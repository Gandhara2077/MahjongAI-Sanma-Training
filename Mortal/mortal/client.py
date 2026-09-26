import prelude

import logging
import socket
import torch
import numpy as np
import time
import gc
import uuid
from os import path
from model import Brain, DQN
from player import TrainPlayer
from common import send_msg, recv_msg
from libriichi.consts import NUM_PLAYERS
from config import config

_DEFAULT_PTS = {
    4: [90, 45, 0, -135],
    3: [90, 0, -90],
}[NUM_PLAYERS]

def main():
    remote = (config['online']['remote']['host'], config['online']['remote']['port'])
    device = torch.device(config['control']['device'])
    version = config['control']['version']
    num_blocks = config['resnet']['num_blocks']
    conv_channels = config['resnet']['conv_channels']

    mortal = Brain(version=version, num_blocks=num_blocks, conv_channels=conv_channels).to(device).eval()
    dqn = DQN(version=version).to(device)
    if config['online']['enable_compile']:
        mortal.compile()
        dqn.compile()

    train_player = TrainPlayer()
    param_version = -1
    session_id = uuid.uuid4().hex
    generated_games = 0
    max_games = int(config.get('online_training', {}).get('max_generated_games', 0))
    if max_games < 0:
        raise ValueError('max_generated_games must not be negative')

    pts = np.array(config.get('test_play', {}).get('pts', _DEFAULT_PTS))
    history_window = config['online']['history_window']
    history = []
    rank_values = np.arange(1, NUM_PLAYERS + 1)

    while True:
        if max_games and generated_games + train_player.seed_count * NUM_PLAYERS > max_games:
            logging.info(f'client game budget reached: {generated_games}/{max_games}')
            return
        while True:
            with socket.socket() as conn:
                conn.connect(remote)
                msg = {
                    'type': 'get_param',
                    'param_version': param_version,
                }
                send_msg(conn, msg)
                rsp = recv_msg(conn, map_location=device)
                if rsp['status'] == 'ok':
                    param_version = rsp['param_version']
                    break
                time.sleep(3)
        mortal.load_state_dict(rsp['mortal'])
        dqn.load_state_dict(rsp['dqn'])
        train_player.sync_opponents(rsp['mortal'], rsp['dqn'], f'{session_id}:{param_version}')
        logging.info('param has been updated')

        rankings, file_list = train_player.train_play(mortal, dqn, device)
        generated_games += train_player.seed_count * NUM_PLAYERS
        avg_rank = rankings @ rank_values / rankings.sum()
        avg_pt = rankings @ pts / rankings.sum()

        history.append(np.array(rankings))
        if len(history) > history_window:
            del history[0]
        sum_rankings = np.sum(history, axis=0)
        ma_avg_rank = sum_rankings @ rank_values / sum_rankings.sum()
        ma_avg_pt = sum_rankings @ pts / sum_rankings.sum()

        logging.info(f'trainee rankings: {rankings} ({avg_rank:.6}, {avg_pt:.6}pt)')
        logging.info(f'last {len(history)} sessions: {sum_rankings} ({ma_avg_rank:.6}, {ma_avg_pt:.6}pt)')

        logs = {}
        for filename in file_list:
            with open(filename, 'rb') as f:
                logs[path.basename(filename)] = f.read()

        with socket.socket() as conn:
            conn.connect(remote)
            send_msg(conn, {
                'type': 'submit_replay',
                'logs': logs,
                'param_version': param_version,
            })
            logging.info('logs have been submitted')
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
