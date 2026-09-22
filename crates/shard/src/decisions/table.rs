use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Action table (loaded from the published artifact, never re-generated).
// ---------------------------------------------------------------------------

/// One template row of `configs/contracts/action_table_v1.json`.
#[derive(Debug, Clone)]
pub struct Template {
    pub kind: String,
    pub tile: Option<u8>,
    pub called_tile: Option<u8>,
    pub consumed: Vec<u8>,
    pub source_offset: Option<i8>,
    pub declares_riichi: bool,
    pub meld_ref_required: bool,
}

/// Published v1 action-table envelope identity (`ACTION_TABLE_ARTIFACT_TYPE`).
pub const ACTION_TABLE_ARTIFACT_TYPE: &str = "hydra2.action_table";
/// Supported table schema (`ACTION_TABLE_SCHEMA_VERSION`, v1 only).
pub const ACTION_TABLE_SCHEMA_VERSION: &str = "1.0.0";
/// Exact template entry keys (`_TEMPLATE_JSON_FIELDS`).
const TEMPLATE_JSON_FIELDS: [&str; 7] = [
    "called_tile",
    "consumed_tiles",
    "declares_riichi",
    "kind",
    "meld_ref_required",
    "source_offset",
    "tile",
];
/// Frozen kind -> ordinal mapping: 13 kinds, fixed 0..12; unknown kind fails closed (`ACTION_KIND_ORDINALS`).
const ACTION_KIND_ORDINALS: [(&str, u32); 13] = [
    ("pass", 0),
    ("discard", 1),
    ("tsumogiri", 2),
    ("riichi_discard", 3),
    ("chi", 4),
    ("pon", 5),
    ("daiminkan", 6),
    ("ankan", 7),
    ("kakan", 8),
    ("ron", 9),
    ("tsumo", 10),
    ("abort_nine_terminals", 11),
    ("accept_abortive_draw", 12),
];

/// Generation-order action table with template -> id lookup.
#[derive(Debug)]
pub struct ActionTable {
    index: HashMap<String, u32>,
    pub len: usize,
    /// Content digest (`table.digest` on the Python `_table` path): sha256
    /// over the RFC 8785 canonical bytes of the digest-free payload
    /// (`{schema_version, actions}` in file order). Binds
    /// `action_table_hash`; byte-equal to Python on the pinned artifact.
    pub digest: String,
}

fn template_key(
    kind: &str,
    tile: Option<u8>,
    called: Option<u8>,
    consumed: &[u8],
    offset: Option<i8>,
    riichi: bool,
    meldref: bool,
) -> String {
    format!(
        "{kind}|{}|{}|{}|{}|{riichi}|{meldref}",
        tile.map(|t| t.to_string()).unwrap_or_default(),
        called.map(|t| t.to_string()).unwrap_or_default(),
        consumed
            .iter()
            .map(|t| t.to_string())
            .collect::<Vec<_>>()
            .join(","),
        offset.map(|o| o.to_string()).unwrap_or_default(),
    )
}

fn kind_ordinal(kind: &str) -> Option<u32> {
    ACTION_KIND_ORDINALS
        .iter()
        .find(|(name, _)| *name == kind)
        .map(|(_, ordinal)| *ordinal)
}

/// Sort key for template generation order.
type TemplateSortKey = (u32, (u8, i64), (u8, i64), Vec<u8>, (u8, i64), bool, bool);
/// Generation order: kind ordinal, then tile/called/consumed/offset with
/// `None` before integers, then riichi/meldref flags (`template_sort_key`).
fn template_sort_key(template: &Template) -> TemplateSortKey {
    fn none_first(value: Option<i64>) -> (u8, i64) {
        match value {
            None => (0, 0),
            Some(v) => (1, v),
        }
    }
    (
        kind_ordinal(&template.kind).unwrap_or(u32::MAX),
        none_first(template.tile.map(|v| v as i64)),
        none_first(template.called_tile.map(|v| v as i64)),
        template.consumed.clone(),
        none_first(template.source_offset.map(|v| v as i64)),
        template.declares_riichi,
        template.meld_ref_required,
    )
}

/// Digest-free template document (`_template_to_json` field mapping).
fn template_json(template: &Template) -> serde_json::Value {
    fn opt_tile(value: Option<u8>) -> serde_json::Value {
        value.map_or(serde_json::Value::Null, |v| {
            serde_json::Value::from(v as u64)
        })
    }
    serde_json::json!({
        "called_tile": opt_tile(template.called_tile),
        "consumed_tiles": template.consumed,
        "declares_riichi": template.declares_riichi,
        "kind": template.kind,
        "meld_ref_required": template.meld_ref_required,
        "source_offset": template.source_offset.map(|v| v as i64).map_or(serde_json::Value::Null, serde_json::Value::from),
        "tile": opt_tile(template.tile),
    })
}

/// Content digest over the digest-free payload in the given template order
/// (`compute_table_digest`).
pub(crate) fn table_content_digest(schema_version: &str, templates: &[Template]) -> String {
    let doc = serde_json::json!({
        "schema_version": schema_version,
        "actions": templates.iter().map(template_json).collect::<Vec<_>>(),
    });
    crate::parity::digest_text(&crate::parity::canonical_json_bytes(&doc))
}

/// Strict template parse (`_template_from_json`): exact key set, known
/// kind, null-or-ranged tile ids (`bool` rejected), ranged offsets, bool
/// flags. Anything else fails closed.
pub(crate) fn parse_template(entry: &serde_json::Value, id: usize) -> Result<Template, String> {
    let obj = entry
        .as_object()
        .ok_or_else(|| format!("action table row {id}: template must be a JSON object"))?;
    if obj.len() != TEMPLATE_JSON_FIELDS.len()
        || !TEMPLATE_JSON_FIELDS
            .iter()
            .all(|key| obj.contains_key(*key))
    {
        return Err(format!(
            "action table row {id}: template entries must hold exactly {TEMPLATE_JSON_FIELDS:?}"
        ));
    }
    let kind = obj
        .get("kind")
        .and_then(|v| v.as_str())
        .filter(|kind| kind_ordinal(kind).is_some())
        .ok_or_else(|| {
            format!(
                "action table row {id}: unknown template kind {:?}",
                obj.get("kind")
            )
        })?;
    let tile = opt_tile_value(obj.get("tile"), id, "tile")?;
    let called_tile = opt_tile_value(obj.get("called_tile"), id, "called_tile")?;
    let consumed_raw = obj
        .get("consumed_tiles")
        .and_then(|v| v.as_array())
        .ok_or_else(|| format!("action table row {id}: consumed_tiles must be an array of ints"))?;
    let mut consumed = Vec::with_capacity(consumed_raw.len());
    for value in consumed_raw {
        consumed.push(
            value
                .as_u64()
                .filter(|v| *v <= 135)
                .and_then(|v| u8::try_from(v).ok())
                .ok_or_else(|| format!("action table row {id}: bad consumed tile"))?,
        );
    }
    let source_offset = match obj.get("source_offset") {
        None | Some(serde_json::Value::Null) => None,
        Some(serde_json::Value::Number(n)) => match n.as_i64() {
            Some(v) if (-1..=2).contains(&v) => Some(i8::try_from(v).unwrap_or(0)),
            _ => return Err(format!("action table row {id}: source_offset invalid: {n}")),
        },
        Some(other) => {
            return Err(format!(
                "action table row {id}: source_offset invalid: {other}"
            ));
        }
    };
    let declares_riichi = obj
        .get("declares_riichi")
        .and_then(|v| v.as_bool())
        .ok_or_else(|| format!("action table row {id}: declares_riichi must be a bool"))?;
    let meld_ref_required = obj
        .get("meld_ref_required")
        .and_then(|v| v.as_bool())
        .ok_or_else(|| format!("action table row {id}: meld_ref_required must be a bool"))?;
    Ok(Template {
        kind: kind.to_string(),
        tile,
        called_tile,
        consumed,
        source_offset,
        declares_riichi,
        meld_ref_required,
    })
}

fn opt_tile_value(
    value: Option<&serde_json::Value>,
    id: usize,
    name: &str,
) -> Result<Option<u8>, String> {
    match value.unwrap_or(&serde_json::Value::Null) {
        serde_json::Value::Null => Ok(None),
        serde_json::Value::Number(n) => n
            .as_u64()
            .filter(|v| *v <= 135)
            .and_then(|v| u8::try_from(v).ok().map(Some))
            .ok_or_else(|| format!("action table row {id}: {name} must be null or int 0..135")),
        other => Err(format!(
            "action table row {id}: {name} must be null or int, got {other}"
        )),
    }
}

#[cfg(test)]
fn opt_u8(value: &serde_json::Value) -> Result<Option<u8>, String> {
    match value {
        serde_json::Value::Null => Ok(None),
        serde_json::Value::Number(n) => n
            .as_u64()
            .filter(|v| *v <= 135)
            .and_then(|v| u8::try_from(v).ok().map(Some))
            .ok_or_else(|| format!("action table tile out of range: {n}")),
        other => Err(format!("action table tile must be int-or-null: {other}")),
    }
}

impl ActionTable {
    /// Load + fully verify the published v1 artifact (6792 rows).
    ///
    /// Mirrors `load_action_table` / `_table_from_document`: envelope
    /// identity, exact payload keys, strict template checks, declared
    /// digest match, and the generation-order rebuild check (a row swap
    /// with an updated digest still fails closed). The verified CONTENT
    /// digest is stored and binds `action_table_hash`.
    pub fn load_json(text: &str) -> Result<Self, String> {
        let doc: serde_json::Value =
            serde_json::from_str(text).map_err(|e| format!("action table parse: {e}"))?;
        let root = doc
            .as_object()
            .ok_or_else(|| "action table: artifact must be a JSON object".to_string())?;
        if root.get("artifact_type").and_then(|v| v.as_str()) != Some(ACTION_TABLE_ARTIFACT_TYPE) {
            return Err(format!(
                "action table: artifact_type must be {ACTION_TABLE_ARTIFACT_TYPE:?}, got {:?}",
                root.get("artifact_type")
            ));
        }
        if root.get("compatibility").and_then(|v| v.as_str()) != Some("exact") {
            return Err(format!(
                "action table: unsupported compatibility {:?}",
                root.get("compatibility")
            ));
        }
        let version = root
            .get("schema_version")
            .and_then(|v| v.as_str())
            .ok_or_else(|| "action table: schema_version must be a string".to_string())?;
        if version.split('.').next() != Some("1") {
            return Err(format!(
                "action table: unknown major schema version {version:?}"
            ));
        }
        if version != ACTION_TABLE_SCHEMA_VERSION {
            return Err(format!(
                "action table: schema_version {version:?} newer than supported {ACTION_TABLE_SCHEMA_VERSION:?}"
            ));
        }
        let payload = root
            .get("payload")
            .and_then(|v| v.as_object())
            .ok_or_else(|| "action table: payload must be an object".to_string())?;
        if payload.len() != 3
            || !["schema_version", "actions", "digest"]
                .iter()
                .all(|key| payload.contains_key(*key))
        {
            return Err(
                "action table: payload must hold exactly schema_version/actions/digest".to_string(),
            );
        }
        if payload.get("schema_version").and_then(|v| v.as_str()) != Some(version) {
            return Err(format!(
                "action table: payload schema_version {:?} != envelope {version:?}",
                payload.get("schema_version")
            ));
        }
        let declared = payload
            .get("digest")
            .and_then(|v| v.as_str())
            .ok_or_else(|| "action table: digest must be a string".to_string())?
            .to_string();
        let actions = payload
            .get("actions")
            .and_then(|v| v.as_array())
            .filter(|arr| !arr.is_empty())
            .ok_or_else(|| "action table: actions must be a non-empty array".to_string())?;
        let mut templates = Vec::with_capacity(actions.len());
        for (id, entry) in actions.iter().enumerate() {
            templates.push(parse_template(entry, id)?);
        }
        let recomputed = table_content_digest(version, &templates);
        if recomputed != declared {
            return Err(format!(
                "action table: declared digest {declared:?} != recomputed {recomputed:?}"
            ));
        }
        // Rebuild check (`build_action_table` sorts then digests): file
        // order must already be generation order.
        let mut ordered = templates.clone();
        ordered.sort_by_key(template_sort_key);
        if table_content_digest(version, &ordered) != declared {
            return Err(
                "action table: templates are not in generation order (rebuilt digest differs)"
                    .to_string(),
            );
        }
        let mut table = ActionTable {
            index: HashMap::new(),
            len: 0,
            digest: declared,
        };
        for (id, template) in templates.iter().enumerate() {
            table.index.insert(
                template_key(
                    &template.kind,
                    template.tile,
                    template.called_tile,
                    &template.consumed,
                    template.source_offset,
                    template.declares_riichi,
                    template.meld_ref_required,
                ),
                u32::try_from(id).map_err(|_| "action table: too many templates".to_string())?,
            );
        }
        if table.index.len() != templates.len() {
            return Err("action table: duplicate action templates are rejected".to_string());
        }
        table.len = table.index.len();
        Ok(table)
    }

    /// Test-only lenient loader: index-only over `payload.actions`, no
    /// envelope or digest checks. Unit-test scaffolding for synthetic
    /// tables; every production path MUST use [`load_json`](Self::load_json).
    #[cfg(test)]
    pub(crate) fn load_unverified_json(text: &str) -> Result<Self, String> {
        let doc: serde_json::Value =
            serde_json::from_str(text).map_err(|e| format!("action table parse: {e}"))?;
        let actions = doc
            .pointer("/payload/actions")
            .and_then(|v| v.as_array())
            .ok_or_else(|| "action table: missing payload.actions".to_string())?;
        let mut index = HashMap::new();
        for (id, entry) in actions.iter().enumerate() {
            let kind = entry
                .get("kind")
                .and_then(|v| v.as_str())
                .ok_or_else(|| format!("action table row {id}: missing kind"))?;
            let tile = opt_u8(entry.get("tile").unwrap_or(&serde_json::Value::Null))?;
            let called = opt_u8(entry.get("called_tile").unwrap_or(&serde_json::Value::Null))?;
            let consumed: Vec<u8> = entry
                .get("consumed_tiles")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .map(|v| {
                            v.as_u64()
                                .filter(|x| *x <= 135)
                                .and_then(|x| u8::try_from(x).ok())
                                .ok_or_else(|| format!("action table row {id}: bad consumed tile"))
                        })
                        .collect::<Result<Vec<u8>, String>>()
                })
                .transpose()?
                .unwrap_or_default();
            let offset = match entry
                .get("source_offset")
                .unwrap_or(&serde_json::Value::Null)
            {
                serde_json::Value::Null => None,
                serde_json::Value::Number(n) => n
                    .as_i64()
                    .filter(|v| (-1..=2).contains(v))
                    .and_then(|v| i8::try_from(v).ok()),
                other => return Err(format!("action table row {id}: bad offset {other}")),
            };
            if offset.is_none()
                && !entry
                    .get("source_offset")
                    .unwrap_or(&serde_json::Value::Null)
                    .is_null()
            {
                return Err(format!("action table row {id}: bad offset"));
            }
            let riichi = entry
                .get("declares_riichi")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            let meldref = entry
                .get("meld_ref_required")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            index.insert(
                template_key(kind, tile, called, &consumed, offset, riichi, meldref),
                u32::try_from(id).unwrap(),
            );
        }
        let len = index.len();
        Ok(ActionTable {
            index,
            len,
            digest: String::new(),
        })
    }

    pub fn lookup(
        &self,
        kind: &str,
        tile: Option<u8>,
        called: Option<u8>,
        consumed: &[u8],
        offset: Option<i8>,
        riichi: bool,
        meldref: bool,
    ) -> Option<u32> {
        self.index
            .get(&template_key(
                kind, tile, called, consumed, offset, riichi, meldref,
            ))
            .copied()
    }
}
