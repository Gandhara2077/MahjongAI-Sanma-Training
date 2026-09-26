use super::{Grp, Invisible};
#[cfg(not(feature = "sanma"))]
use crate::chi_type::ChiType;
#[cfg(feature = "sanma")]
use crate::consts::ACTION_NUKIDORA;
use crate::consts::{
    ACTION_AGARI, ACTION_KAN, ACTION_PASS, ACTION_PON, ACTION_RIICHI, ACTION_RYUKYOKU, NUM_PLAYERS,
};
#[cfg(not(feature = "sanma"))]
use crate::consts::{ACTION_CHI_HIGH, ACTION_CHI_LOW, ACTION_CHI_MID};
use crate::mjai::Event;
use crate::state::PlayerState;
use std::array;
use std::fs::File;
use std::io;
use std::mem;

use ahash::AHashSet;
use anyhow::{Context, Result, bail};
use derivative::Derivative;
use flate2::read::GzDecoder;
use ndarray::prelude::*;
use numpy::{PyArray1, PyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;
use serde_json as json;
use tinyvec::ArrayVec;

#[pyclass]
#[derive(Derivative)]
#[derivative(Debug)]
pub struct GameplayLoader {
    #[pyo3(get)]
    version: u32,
    #[pyo3(get)]
    oracle: bool,
    #[pyo3(get)]
    player_names: Vec<String>,
    #[pyo3(get)]
    excludes: Vec<String>,
    #[pyo3(get)]
    trust_seed: bool,
    #[pyo3(get)]
    always_include_kan_select: bool,
    #[pyo3(get)]
    augmented: bool,
    #[pyo3(get)]
    encode_observations: bool,

    #[derivative(Debug = "ignore")]
    player_names_set: AHashSet<String>,
    #[derivative(Debug = "ignore")]
    excludes_set: AHashSet<String>,
}

#[pyclass]
#[derive(Clone, Default)]
pub struct Gameplay {
    // per move
    pub obs: Vec<Array2<f32>>,
    pub invisible_obs: Vec<Array2<f32>>,
    pub actions: Vec<i64>,
    pub masks: Vec<Array1<bool>>,
    pub at_kyoku: Vec<u8>,
    pub dones: Vec<bool>,
    pub apply_gamma: Vec<bool>,
    pub at_turns: Vec<u8>,
    pub shantens: Vec<i8>,
    // Sample source tracking, needed by the sanma MahjongCopilot compat
    // replay (additive for yonma builds).
    pub event_indices: Vec<usize>,
    pub at_kan_select: Vec<bool>,

    // per game
    pub grp: Grp, // actually per kyoku though
    pub player_id: u8,
    pub player_name: String,
}

struct LoaderContext<'a> {
    config: &'a GameplayLoader,
    invisibles: Option<&'a [Invisible]>,

    state: PlayerState,
    kyoku_idx: usize,

    // fields below are only used for oracle
    opponent_states: [PlayerState; NUM_PLAYERS - 1],
    from_rinshan: bool,
    yama_idx: usize,
    rinshan_idx: usize,
}

#[pymethods]
impl GameplayLoader {
    #[new]
    #[pyo3(signature = (
        version,
        *,
        oracle = true,
        player_names = None,
        excludes = None,
        trust_seed = false,
        always_include_kan_select = true,
        augmented = false,
        encode_observations = true,
    ))]
    fn new(
        version: u32,
        oracle: bool,
        player_names: Option<Vec<String>>,
        excludes: Option<Vec<String>>,
        trust_seed: bool,
        always_include_kan_select: bool,
        augmented: bool,
        encode_observations: bool,
    ) -> Self {
        let player_names = player_names.unwrap_or_default();
        let player_names_set = player_names.iter().cloned().collect();
        let excludes = excludes.unwrap_or_default();
        let excludes_set = excludes.iter().cloned().collect();
        Self {
            version,
            oracle,
            player_names,
            excludes,
            trust_seed,
            always_include_kan_select,
            augmented,
            encode_observations,
            player_names_set,
            excludes_set,
        }
    }

    // Nested result is too hard to handle...
    fn load_log(&self, raw_log: &str) -> Result<Vec<Gameplay>> {
        let mut events = raw_log
            .lines()
            .map(json::from_str)
            .collect::<Result<Vec<Event>, _>>()
            .context("failed to parse log")?;
        if self.augmented {
            events.iter_mut().for_each(Event::augment);
        }
        self.load_events(&events)
    }

    #[pyo3(name = "load_gz_log_files")]
    fn load_gz_log_files_py(&self, gzip_filenames: Vec<String>) -> Result<Vec<Vec<Gameplay>>> {
        self.load_gz_log_files(gzip_filenames)
    }

    fn __repr__(&self) -> String {
        format!("{self:?}")
    }
}

impl GameplayLoader {
    pub fn load_gz_log_files<V, S>(&self, gzip_filenames: V) -> Result<Vec<Vec<Gameplay>>>
    where
        V: IntoParallelIterator<Item = S>,
        S: AsRef<str>,
    {
        gzip_filenames
            .into_par_iter()
            .map(|f| {
                let filename = f.as_ref();
                let inner = || {
                    let file = File::open(filename)?;
                    let gz = GzDecoder::new(file);
                    let raw = io::read_to_string(gz)?;
                    self.load_log(&raw)
                };
                inner().with_context(|| format!("error when reading {filename}"))
            })
            .collect()
    }

    pub fn load_events(&self, events: &[Event]) -> Result<Vec<Gameplay>> {
        let invisibles = self.oracle.then(|| Invisible::new(events, self.trust_seed));

        let [Event::StartGame { names, .. }, ..] = events else {
            bail!("empty or invalid game log");
        };
        names
            .iter()
            .enumerate()
            .filter(|&(_, name)| {
                if !self.player_names_set.is_empty() {
                    return self.player_names_set.contains(name);
                }
                if !self.excludes_set.is_empty() {
                    return !self.excludes_set.contains(name);
                }
                true
            })
            .map(|(i, _)| i as u8)
            .collect::<ArrayVec<[_; NUM_PLAYERS]>>()
            .into_par_iter()
            .map(|&player_id| {
                Gameplay::load_events_by_player(self, events, player_id, invisibles.as_deref())
            })
            .collect()
    }
}

#[pymethods]
impl Gameplay {
    fn take_obs<'py>(&mut self, py: Python<'py>) -> Vec<Bound<'py, PyArray2<f32>>> {
        mem::take(&mut self.obs)
            .into_iter()
            .map(|v| PyArray2::from_owned_array(py, v))
            .collect()
    }
    fn take_invisible_obs<'py>(&mut self, py: Python<'py>) -> Vec<Bound<'py, PyArray2<f32>>> {
        mem::take(&mut self.invisible_obs)
            .into_iter()
            .map(|v| PyArray2::from_owned_array(py, v))
            .collect()
    }
    fn take_actions(&mut self) -> Vec<i64> {
        mem::take(&mut self.actions)
    }
    fn take_masks<'py>(&mut self, py: Python<'py>) -> Vec<Bound<'py, PyArray1<bool>>> {
        mem::take(&mut self.masks)
            .into_iter()
            .map(|v| PyArray1::from_owned_array(py, v))
            .collect()
    }
    fn take_at_kyoku(&mut self) -> Vec<u8> {
        mem::take(&mut self.at_kyoku)
    }
    fn take_dones(&mut self) -> Vec<bool> {
        mem::take(&mut self.dones)
    }
    fn take_apply_gamma(&mut self) -> Vec<bool> {
        mem::take(&mut self.apply_gamma)
    }
    fn take_at_turns(&mut self) -> Vec<u8> {
        mem::take(&mut self.at_turns)
    }
    fn take_shantens(&mut self) -> Vec<i8> {
        mem::take(&mut self.shantens)
    }
    fn take_event_indices(&mut self) -> Vec<usize> {
        mem::take(&mut self.event_indices)
    }
    fn take_at_kan_select(&mut self) -> Vec<bool> {
        mem::take(&mut self.at_kan_select)
    }

    fn take_grp(&mut self) -> Grp {
        mem::take(&mut self.grp)
    }

    const fn take_player_id(&self) -> u8 {
        self.player_id
    }
}

impl Gameplay {
    fn load_events_by_player(
        config: &GameplayLoader,
        events: &[Event],
        player_id: u8,
        invisibles: Option<&[Invisible]>,
    ) -> Result<Self> {
        let grp = Grp::load_events(events)?;

        let mut data = Self {
            grp,
            player_id,
            ..Default::default()
        };

        let mut ctx = LoaderContext {
            config,
            invisibles,
            state: PlayerState::new(player_id),
            kyoku_idx: 0,
            // end_state: EndState::Passive,
            opponent_states: array::from_fn(|i| {
                PlayerState::new((player_id + i as u8 + 1) % NUM_PLAYERS as u8)
            }),
            from_rinshan: false,
            yama_idx: 0,
            rinshan_idx: 0,
        };

        // It is guaranteed that there are at least NUM_PLAYERS events.
        // tsumo/dahai -> ryukyoku/hora -> end kyoku (yonma: -> end game).
        // The window spans cur + (NUM_PLAYERS - 1) lookaheads, i.e. the max
        // number of consecutive Hora events a single discard can trigger
        // (triple ron in yonma, double ron in sanma).
        for (event_index, wnd) in events.windows(NUM_PLAYERS).enumerate() {
            data.extend_from_event_window(&mut ctx, event_index, wnd.try_into().unwrap())?;
        }

        data.dones = data.at_kyoku.windows(2).map(|w| w[1] > w[0]).collect();
        data.dones.push(true);

        Ok(data)
    }

    fn extend_from_event_window(
        &mut self,
        ctx: &mut LoaderContext<'_>,
        event_index: usize,
        wnd: &[Event; NUM_PLAYERS],
    ) -> Result<()> {
        let LoaderContext {
            config,
            invisibles,
            state,
            kyoku_idx,
            opponent_states,
            from_rinshan,
            yama_idx,
            rinshan_idx,
        } = ctx;

        let cur = &wnd[0];
        let next = if matches!(wnd[1], Event::ReachAccepted { .. } | Event::Dora { .. }) {
            &wnd[2]
        } else {
            &wnd[1]
        };

        match cur {
            Event::StartGame { names, .. } => {
                self.player_name.clone_from(&names[self.player_id as usize]);
            }
            Event::EndKyoku => *kyoku_idx += 1,
            _ => (),
        }

        if invisibles.is_some() {
            match cur {
                Event::EndKyoku => {
                    *from_rinshan = false;
                    *yama_idx = 0;
                    *rinshan_idx = 0;
                }
                Event::Tsumo { .. } => {
                    if *from_rinshan {
                        *rinshan_idx += 1;
                        *from_rinshan = false;
                    } else {
                        *yama_idx += 1;
                    }
                }
                Event::Ankan { .. } | Event::Kakan { .. } | Event::Daiminkan { .. } => {
                    *from_rinshan = true;
                }
                _ => (),
            };

            for s in opponent_states {
                s.update(cur)?;
            }
        }

        let cans = state.update(cur)?;
        if !cans.can_act() {
            return Ok(());
        }

        let mut kan_select = None;
        let label_opt = match *next {
            Event::Dahai { pai, .. } => Some(pai.as_usize()),
            Event::Reach { .. } => Some(ACTION_RIICHI),
            #[cfg(not(feature = "sanma"))]
            Event::Chi {
                actor,
                pai,
                consumed,
                ..
            } if actor == self.player_id => match ChiType::new(consumed, pai) {
                ChiType::Low => Some(ACTION_CHI_LOW),
                ChiType::Mid => Some(ACTION_CHI_MID),
                ChiType::High => Some(ACTION_CHI_HIGH),
            },
            // Sanma: chi is disabled. If chi events somehow appear (legacy
            // yonma logs), skip the entry rather than emitting a label.
            #[cfg(feature = "sanma")]
            Event::Chi { actor, .. } if actor == self.player_id => None,
            #[cfg(feature = "sanma")]
            Event::Nukidora { actor, .. } if actor == self.player_id => Some(ACTION_NUKIDORA),
            Event::Pon { actor, .. } if actor == self.player_id => Some(ACTION_PON),
            #[cfg(not(feature = "sanma"))]
            Event::Daiminkan { actor, pai, .. } if actor == self.player_id => {
                if config.always_include_kan_select {
                    kan_select = Some(pai.deaka().as_usize());
                }
                Some(ACTION_KAN)
            }
            // Sanma: a daiminkan's tile is fully determined by the called
            // tile, so no redundant kan-select sample is emitted.
            #[cfg(feature = "sanma")]
            Event::Daiminkan { actor, .. } if actor == self.player_id => Some(ACTION_KAN),
            Event::Kakan { pai, .. } => {
                if config.always_include_kan_select || state.kakan_candidates().len() > 1 {
                    kan_select = Some(pai.deaka().as_usize());
                }
                Some(ACTION_KAN)
            }
            Event::Ankan { consumed, .. } => {
                if config.always_include_kan_select || state.ankan_candidates().len() > 1 {
                    kan_select = Some(consumed[0].deaka().as_usize());
                }
                Some(ACTION_KAN)
            }
            Event::Ryukyoku { .. } if cans.can_ryukyoku => Some(ACTION_RYUKYOKU),
            _ => {
                let mut ret = None;

                let has_any_ron = matches!(wnd[1], Event::Hora { .. });
                if has_any_ron {
                    // Check if the POV is one of those who made Hora.
                    for ev in &wnd[1..] {
                        match *ev {
                            Event::EndKyoku => break,
                            Event::Hora { actor, .. } if actor == self.player_id => {
                                ret = Some(ACTION_AGARI);
                                break;
                            }
                            _ => (),
                        };
                    }
                }

                if ret.is_none() {
                    // It is now proven there is no ron from the POV.
                    #[cfg(not(feature = "sanma"))]
                    if cans.can_chi() && matches!(next, Event::Tsumo { .. })
                        || (cans.can_pon || cans.can_daiminkan || cans.can_ron_agari)
                            && !has_any_ron
                    {
                        // Can chi, but actively denied instead of being
                        // interrupted by other's pon/daiminkan/ron.
                        //
                        // or
                        //
                        // Can pon/daiminkan/ron, but actively denied
                        // instead of being interrupted by other's ron.
                        ret = Some(ACTION_PASS);
                    }
                    // Sanma: no chi. We still keep the original "deny"
                    // semantics for pon/daiminkan/ron.
                    #[cfg(feature = "sanma")]
                    if (cans.can_pon || cans.can_daiminkan || cans.can_ron_agari) && !has_any_ron {
                        // Can pon/daiminkan/ron, but actively denied
                        // instead of being interrupted by other's ron.
                        ret = Some(ACTION_PASS);
                    }
                }

                ret
            }
        };

        if let Some(label) = label_opt {
            self.add_entry(ctx, event_index, false, label);
            if let Some(kan) = kan_select {
                self.add_entry(ctx, event_index, true, kan);
            }
        }
        Ok(())
    }

    fn add_entry(
        &mut self,
        ctx: &LoaderContext<'_>,
        event_index: usize,
        at_kan_select: bool,
        label: usize,
    ) {
        if ctx.config.encode_observations {
            let (feature, mask) = ctx.state.encode_obs(ctx.config.version, at_kan_select);
            self.obs.push(feature);
            self.masks.push(mask);
        }
        self.actions.push(label as i64);
        self.at_kyoku.push(ctx.kyoku_idx as u8);
        // only discard and kan will discount
        self.apply_gamma.push(label <= ACTION_RIICHI);
        self.at_turns.push(ctx.state.at_turn());
        self.shantens.push(ctx.state.shanten());
        self.event_indices.push(event_index);
        self.at_kan_select.push(at_kan_select);

        if let Some(invisibles) = ctx.invisibles {
            let invisible_obs = invisibles[ctx.kyoku_idx].encode(
                &ctx.opponent_states,
                ctx.yama_idx,
                ctx.rinshan_idx,
                ctx.config.version,
            );
            self.invisible_obs.push(invisible_obs);
        }
    }
}

#[cfg(all(test, feature = "sanma"))]
mod tests {
    use super::*;
    use crate::consts::MAX_VERSION;

    #[test]
    fn daiminkan_does_not_emit_a_redundant_kan_selection_sample() {
        let raw_log = r#"{"type":"start_game","names":["p0","p1","p2"]}
{"type":"start_kyoku","bakaze":"E","dora_marker":"1s","kyoku":1,"honba":0,"kyotaku":0,"oya":0,"scores":[35000,35000,35000],"tehais":[["1m","9m","1p","2p","3p","4p","5p","6p","6p","6p","7p","8p","E"],["?","?","?","?","?","?","?","?","?","?","?","?","?"],["?","?","?","?","?","?","?","?","?","?","?","?","?"]]}
{"type":"tsumo","actor":0,"pai":"S"}
{"type":"dahai","actor":0,"pai":"S","tsumogiri":true}
{"type":"tsumo","actor":1,"pai":"9s"}
{"type":"dahai","actor":1,"pai":"6p","tsumogiri":false}
{"type":"daiminkan","actor":0,"target":1,"pai":"6p","consumed":["6p","6p","6p"]}
{"type":"tsumo","actor":0,"pai":"7s"}
{"type":"dahai","actor":0,"pai":"7s","tsumogiri":true}
{"type":"ryukyoku","deltas":[0,0,0]}
{"type":"end_kyoku"}
{"type":"end_game"}"#;
        let loader = GameplayLoader::new(
            MAX_VERSION,
            false,
            Some(vec!["p0".to_owned()]),
            None,
            false,
            true,
            false,
            false,
        );
        let mut games = loader.load_log(raw_log).unwrap();
        let game = games.pop().unwrap();
        let samples = game
            .event_indices
            .iter()
            .zip(&game.actions)
            .zip(&game.at_kan_select)
            .filter_map(|((&event_index, &action), &at_kan_select)| {
                (event_index == 5).then_some((action, at_kan_select))
            })
            .collect::<Vec<_>>();

        assert_eq!(samples, vec![(ACTION_KAN as i64, false)]);
    }

    #[test]
    fn metadata_only_loader_skips_native_observation_tensors() {
        let raw_log = r#"{"type":"start_game","names":["p0","p1","p2"]}
{"type":"start_kyoku","bakaze":"E","dora_marker":"1s","kyoku":1,"honba":0,"kyotaku":0,"oya":0,"scores":[35000,35000,35000],"tehais":[["1m","9m","1p","2p","3p","4p","5p","6p","6p","6p","7p","8p","E"],["?","?","?","?","?","?","?","?","?","?","?","?","?"],["?","?","?","?","?","?","?","?","?","?","?","?","?"]]}
{"type":"tsumo","actor":0,"pai":"S"}
{"type":"dahai","actor":0,"pai":"S","tsumogiri":true}
{"type":"ryukyoku","deltas":[0,0,0]}
{"type":"end_kyoku"}
{"type":"end_game"}"#;
        let loader = GameplayLoader::new(
            MAX_VERSION,
            false,
            Some(vec!["p0".to_owned()]),
            None,
            false,
            true,
            false,
            false,
        );
        let mut games = loader.load_log(raw_log).unwrap();
        let game = games.pop().unwrap();
        assert!(!game.actions.is_empty());
        assert!(game.obs.is_empty());
        assert!(game.masks.is_empty());
        assert_eq!(game.actions.len(), game.event_indices.len());
        assert_eq!(game.actions.len(), game.at_kan_select.len());
    }
}
