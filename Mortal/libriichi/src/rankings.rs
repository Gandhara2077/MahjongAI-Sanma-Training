use crate::consts::NUM_PLAYERS;

#[derive(Debug, Clone, Copy)]
pub struct Rankings {
    pub player_by_rank: [u8; NUM_PLAYERS],
    pub rank_by_player: [u8; NUM_PLAYERS],
}

impl Rankings {
    pub fn new(scores: [i32; NUM_PLAYERS]) -> Self {
        let mut player_by_rank = std::array::from_fn(|i| i as u8);
        player_by_rank.sort_by_key(|&i| -scores[i as usize]);

        let mut rank_by_player = [0; NUM_PLAYERS];
        for (rank, id) in player_by_rank.into_iter().enumerate() {
            rank_by_player[id as usize] = rank as u8;
        }

        Self {
            player_by_rank,
            rank_by_player,
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    #[cfg(not(feature = "sanma"))]
    fn rankings() {
        let mut scores = [25000, 25000, 30000, 20000];

        let rk = Rankings::new(scores);
        assert_eq!(rk.player_by_rank, [2, 0, 1, 3]);
        assert_eq!(rk.rank_by_player, [1, 2, 0, 3]);
        *scores.iter_mut().min_by_key(|s| -**s).unwrap() = 0;
        assert_eq!(scores, [25000, 25000, 0, 20000]);

        scores = [25000, 25000, 25000, 25000];
        let rk = Rankings::new(scores);
        assert_eq!(rk.player_by_rank, [0, 1, 2, 3]);
        assert_eq!(rk.rank_by_player, [0, 1, 2, 3]);
        *scores.iter_mut().min_by_key(|s| -**s).unwrap() = 0;
        assert_eq!(scores, [0, 25000, 25000, 25000]);

        scores = [18000, 32000, 32000, 18000];
        let rk = Rankings::new(scores);
        assert_eq!(rk.player_by_rank, [1, 2, 0, 3]);
        assert_eq!(rk.rank_by_player, [2, 0, 1, 3]);
        *scores.iter_mut().min_by_key(|s| -**s).unwrap() = 0;
        assert_eq!(scores, [18000, 0, 32000, 18000]);

        scores = [32000, 18000, 18000, 32000];
        let rk = Rankings::new(scores);
        assert_eq!(rk.player_by_rank, [0, 3, 1, 2]);
        assert_eq!(rk.rank_by_player, [0, 2, 3, 1]);
        *scores.iter_mut().min_by_key(|s| -**s).unwrap() = 0;
        assert_eq!(scores, [0, 18000, 18000, 32000]);

        scores = [0, 100000, 0, 0];
        let rk = Rankings::new(scores);
        assert_eq!(rk.player_by_rank, [1, 0, 2, 3]);
        assert_eq!(rk.rank_by_player, [1, 0, 2, 3]);
        *scores.iter_mut().min_by_key(|s| -**s).unwrap() = 0;
        assert_eq!(scores, [0; 4]);
    }

    #[test]
    #[cfg(feature = "sanma")]
    fn rankings() {
        let scores = [35000, 35000, 30000];
        let rk = Rankings::new(scores);
        assert_eq!(rk.player_by_rank, [0, 1, 2]);
        assert_eq!(rk.rank_by_player, [0, 1, 2]);

        let scores = [20000, 40000, 40000];
        let rk = Rankings::new(scores);
        assert_eq!(rk.player_by_rank, [1, 2, 0]);
        assert_eq!(rk.rank_by_player, [2, 0, 1]);
    }
}
