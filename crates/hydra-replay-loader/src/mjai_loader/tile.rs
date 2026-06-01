use super::*;

#[inline]
pub fn tile136_to_type(tile136: u8) -> u8 {
    tile136 / 4
}

pub(super) fn mjai_tile(tile: &str) -> io::Result<u8> {
    mjai_to_tid(tile).ok_or_else(|| invalid_data(format!("invalid mjai tile: {tile}")))
}

pub(super) fn mjai_tile_type(tile: &str) -> io::Result<u8> {
    Ok(tile136_to_type(mjai_tile(tile)?))
}

pub(super) fn rel_opp(observer: usize, actor: usize) -> Option<usize> {
    let idx = ((actor + 4 - observer) % 4).wrapping_sub(1);
    (idx < 3).then_some(idx)
}

pub(super) fn abs_opp(observer: usize, rel: usize) -> usize {
    (observer + rel + 1) % 4
}

pub fn update_safety(safety: &mut [SafetyInfo; 4], event: &MjaiEvent) -> io::Result<()> {
    match event {
        MjaiEvent::StartKyoku { dora_marker, .. } => {
            *safety = array::from_fn(|_| SafetyInfo::default());
            let dora = mjai_tile_type(dora_marker)?;
            for info in safety.iter_mut() {
                info.on_dora_revealed(dora);
            }
        }
        MjaiEvent::Dora { dora_marker } => {
            let dora = mjai_tile_type(dora_marker)?;
            for info in safety.iter_mut() {
                info.on_dora_revealed(dora);
            }
        }
        MjaiEvent::Reach { actor } => {
            for (observer, info) in safety.iter_mut().enumerate() {
                if observer != *actor
                    && let Some(opp) = rel_opp(observer, *actor)
                {
                    info.on_riichi(opp);
                }
            }
        }
        MjaiEvent::Dahai {
            actor,
            pai,
            tsumogiri,
        } => {
            let tile = mjai_tile_type(pai)?;
            for (observer, info) in safety.iter_mut().enumerate() {
                if observer != *actor
                    && let Some(opp) = rel_opp(observer, *actor)
                {
                    info.on_discard(tile, opp, !*tsumogiri);
                }
            }
        }
        MjaiEvent::Pon {
            actor, consumed, ..
        }
        | MjaiEvent::Chi {
            actor, consumed, ..
        }
        | MjaiEvent::Kan {
            actor, consumed, ..
        }
        | MjaiEvent::Ankan { actor, consumed } => {
            let tiles = consumed
                .iter()
                .map(|tile| mjai_tile_type(tile))
                .collect::<io::Result<Vec<_>>>()?;
            for (observer, info) in safety.iter_mut().enumerate() {
                if observer != *actor && rel_opp(observer, *actor).is_some() {
                    info.on_call(&tiles);
                }
            }
        }
        MjaiEvent::Kakan { actor, pai } => {
            let tiles = [mjai_tile_type(pai)?];
            for (observer, info) in safety.iter_mut().enumerate() {
                if observer != *actor && rel_opp(observer, *actor).is_some() {
                    info.on_call(&tiles);
                }
            }
        }
        _ => {}
    }
    Ok(())
}

pub fn next_discards_after(events: &[MjaiEvent]) -> io::Result<Vec<[Option<u8>; 4]>> {
    let mut out = vec![[None; 4]; events.len()];
    let mut next = [None; 4];
    for (idx, event) in events.iter().enumerate().rev() {
        out[idx] = next;
        if let MjaiEvent::Dahai { actor, pai, .. } = event {
            next[*actor] = Some(mjai_tile_type(pai)?);
        }
    }
    Ok(out)
}
