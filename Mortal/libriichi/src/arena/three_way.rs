use super::game::{BatchGame, Index};
use super::result::GameResult;
use crate::agent::{BatchAgent, new_py_agent};
use std::array;
use std::fs::{self, File};
use std::io;
use std::iter;
use std::path::PathBuf;
use std::time::Duration;

use anyhow::{Result, ensure};
use flate2::Compression;
use flate2::read::GzEncoder;
use indicatif::{ParallelProgressIterator, ProgressBar, ProgressStyle};
use pyo3::prelude::*;
use rayon::prelude::*;

const MODEL_IDX_PER_SPLIT: [[usize; 3]; 3] = [
    [0, 1, 2], // [32k, 36k, 40k]
    [1, 2, 0], // [36k, 40k, 32k]
    [2, 0, 1], // [40k, 32k, 36k]
];

fn indexes_for_seed(player_id_starts: [usize; 3]) -> [[Index; 3]; 3] {
    let mut next_player_id = player_id_starts;
    MODEL_IDX_PER_SPLIT.map(|model_idxs| {
        model_idxs.map(|agent_idx| {
            let index = Index {
                agent_idx,
                player_id_idx: next_player_id[agent_idx],
            };
            next_player_id[agent_idx] += 1;
            index
        })
    })
}

fn player_ids_for_seeds(seed_count: u64) -> [Vec<u8>; 3] {
    let mut player_ids = array::from_fn(|_| Vec::with_capacity(seed_count as usize * 3));
    for _ in 0..seed_count {
        for model_idxs in MODEL_IDX_PER_SPLIT {
            for (seat, model_idx) in model_idxs.into_iter().enumerate() {
                player_ids[model_idx].push(seat as u8);
            }
        }
    }
    player_ids
}

#[pyclass]
#[derive(Clone, Default)]
pub struct ThreeWay {
    pub disable_progress_bar: bool,
    pub log_dir: Option<String>,
}

#[pymethods]
impl ThreeWay {
    #[new]
    #[pyo3(signature = (*, disable_progress_bar=false, log_dir=None))]
    const fn new(disable_progress_bar: bool, log_dir: Option<String>) -> Self {
        Self {
            disable_progress_bar,
            log_dir,
        }
    }

    /// Returns each model's zero-based rank for every completed game.
    pub fn py_vs_py(
        &self,
        engines: Vec<PyObject>,
        seed_start: (u64, u64),
        seed_count: u64,
        py: Python<'_>,
    ) -> Result<Vec<[u8; 3]>> {
        ensure!(engines.len() == 3, "expected exactly three engines");
        let mut engines = engines.into_iter();
        let engine0 = engines.next().unwrap();
        let engine1 = engines.next().unwrap();
        let engine2 = engines.next().unwrap();

        py.allow_threads(move || {
            let results = self.run_batch(
                |player_ids| new_py_agent(engine0, player_ids),
                |player_ids| new_py_agent(engine1, player_ids),
                |player_ids| new_py_agent(engine2, player_ids),
                seed_start,
                seed_count,
            )?;

            Ok(results
                .into_iter()
                .enumerate()
                .map(|(i, result)| {
                    let rank_by_seat = result.rankings().rank_by_player;
                    let mut rank_by_model = [0; 3];
                    for (seat, model_idx) in MODEL_IDX_PER_SPLIT[i % 3].into_iter().enumerate() {
                        rank_by_model[model_idx] = rank_by_seat[seat];
                    }
                    rank_by_model
                })
                .collect())
        })
    }
}

impl ThreeWay {
    pub fn run_batch<C0, C1, C2>(
        &self,
        new_agent0: C0,
        new_agent1: C1,
        new_agent2: C2,
        seed_start: (u64, u64),
        seed_count: u64,
    ) -> Result<Vec<GameResult>>
    where
        C0: FnOnce(&[u8]) -> Result<Box<dyn BatchAgent>>,
        C1: FnOnce(&[u8]) -> Result<Box<dyn BatchAgent>>,
        C2: FnOnce(&[u8]) -> Result<Box<dyn BatchAgent>>,
    {
        ensure!(seed_count > 0, "seed_count must be positive");
        if let Some(dir) = &self.log_dir {
            fs::create_dir_all(dir)?;
        }

        log::info!(
            "seed: [{}, {}) w/ {:#x}, start {} sets, {} hanchans",
            seed_start.0,
            seed_start.0 + seed_count,
            seed_start.1,
            seed_count,
            seed_count * 3,
        );

        let seeds: Vec<_> = (seed_start.0..seed_start.0 + seed_count)
            .flat_map(|seed| iter::repeat_n((seed, seed_start.1), 3))
            .collect();

        let player_ids = player_ids_for_seeds(seed_count);
        let mut agents = [
            new_agent0(&player_ids[0])?,
            new_agent1(&player_ids[1])?,
            new_agent2(&player_ids[2])?,
        ];

        let mut indexes = Vec::with_capacity(seed_count as usize * 3);
        for seed_idx in 0..seed_count {
            let player_id_start = [seed_idx as usize * 3; 3];
            indexes.extend(indexes_for_seed(player_id_start));
        }

        let batch_game = BatchGame::tenhou_hanchan(self.disable_progress_bar);
        let results = batch_game.run(&mut agents, &indexes, &seeds)?;

        if let Some(dir) = &self.log_dir {
            log::info!("dumping game logs");

            let bar = if self.disable_progress_bar {
                ProgressBar::hidden()
            } else {
                ProgressBar::new(seed_count * 3)
            };
            const TEMPLATE: &str = "[{elapsed_precise}] [{wide_bar}] {pos}/{len} {percent:>3}%";
            bar.set_style(ProgressStyle::with_template(TEMPLATE)?.progress_chars("#-"));
            bar.enable_steady_tick(Duration::from_millis(150));

            results
                .par_iter()
                .progress_with(bar)
                .enumerate()
                .try_for_each(|(i, game_result)| {
                    let split_name = ["a", "b", "c"][i % 3];
                    let (seed, key) = game_result.seed;
                    let filename: PathBuf = [dir, &format!("{seed}_{key}_{split_name}.json.gz")]
                        .iter()
                        .collect();

                    let log = game_result.dump_json_log()?;
                    let mut comp = GzEncoder::new(log.as_bytes(), Compression::best());
                    let mut f = File::create(filename)?;
                    io::copy(&mut comp, &mut f)?;

                    anyhow::Ok(())
                })?;
        }

        Ok(results)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn rotates_each_model_through_each_seat() {
        let actual = indexes_for_seed([0, 0, 0])
            .map(|split| split.map(|index| [index.agent_idx, index.player_id_idx]));
        assert_eq!(
            actual,
            [
                [[0, 0], [1, 0], [2, 0]],
                [[1, 1], [2, 1], [0, 1]],
                [[2, 2], [0, 2], [1, 2]],
            ]
        );
    }
}
