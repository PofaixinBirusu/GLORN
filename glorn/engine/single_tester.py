from typing import Dict

import torch
from tqdm import tqdm

from glorn.engine.base_tester import BaseTester
from glorn.utils.summary_board import SummaryBoard
from glorn.utils.timer import Timer
from glorn.utils.common import get_log_string
from glorn.utils.torch import release_cuda, to_cuda


class SingleTester(BaseTester):
    def __init__(self, cfg, parser=None, cudnn_deterministic=True):
        super().__init__(cfg, parser=parser, cudnn_deterministic=cudnn_deterministic)

    def before_test_epoch(self):
        pass

    def before_test_step(self, iteration, data_dict):
        pass

    def test_step(self, iteration, data_dict) -> Dict:
        pass

    def eval_step(self, iteration, data_dict, output_dict) -> Dict:
        pass

    def after_test_step(self, iteration, data_dict, output_dict, result_dict):
        pass

    def after_test_epoch(self):
        pass

    def summary_string(self, iteration, data_dict, output_dict, result_dict):
        return get_log_string(result_dict)

    def run(self):
        assert self.test_loader is not None
        self.load_snapshot(self.args.snapshot)
        self.model.eval()
        torch.set_grad_enabled(False)
        self.before_test_epoch()
        summary_board = SummaryBoard(adaptive=True)
        timer = Timer()
        total_iterations = len(self.test_loader)
        pbar = tqdm(enumerate(self.test_loader), total=total_iterations)
        for iteration, data_dict in pbar:
            # on start
            self.iteration = iteration + 1
            # if self.iteration in [167, 168, 607, 803, 989, 995, 1003, 1018, 1022, 1125, 1165, 1168, 1428, 1429, 1430, 1432, 1433, 1434, 1435, 1438, 1449, 1450]:
            #     continue
            # if self.iteration in [1297, 1599, 1630, 1631, 1634]:
            #     continue
            data_dict = to_cuda(data_dict)
            self.before_test_step(self.iteration, data_dict)
            # test step
            torch.cuda.synchronize()
            timer.add_prepare_time()
            output_dict = self.test_step(self.iteration, data_dict)
            torch.cuda.synchronize()
            timer.add_process_time()
            # eval step
            try:
                result_dict = self.eval_step(self.iteration, data_dict, output_dict)
            except Exception as e:
                print(e)
            # after step
            self.after_test_step(self.iteration, data_dict, output_dict, result_dict)
            # logging
            result_dict = release_cuda(result_dict)
            summary_board.update_from_result_dict(result_dict)
            message = self.summary_string(self.iteration, data_dict, output_dict, result_dict)
            message += f', {timer.tostring()}'
            pbar.set_description(message)
            torch.cuda.empty_cache()
        self.after_test_epoch()
        summary_dict = summary_board.summary()
        message = get_log_string(result_dict=summary_dict, timer=timer)
        self.logger.critical(message)
