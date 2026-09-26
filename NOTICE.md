# Attribution and Distribution

This repository contains a modified copy of
[Equim-chan/Mortal](https://github.com/Equim-chan/Mortal).
Mortal and derivative source are distributed under the GNU Affero General
Public License, version 3; see [the full license](Mortal/LICENSE). Preserve
upstream copyright and license notices when redistributing this project.

Model weights, downloaded Tenhou game logs, and the historical MahjongCopilot
`libriichi3p` reference binary are external inputs, not part of this source
distribution. Their presence on a developer machine does not grant permission
to redistribute them. Obtain them lawfully and review their respective terms
before publishing data, binaries, or trained weights. Do not upload the local
`baselines/` weights merely because the source code is public.

The training code loads PyTorch checkpoints. Only load checkpoints from trusted
sources; do not disable restricted loading to make an unknown checkpoint work.
Online training services are intended for loopback/local trusted processes,
not exposure to the public Internet.
