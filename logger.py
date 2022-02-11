import os
import logging
import sys

class txt_logger:
    def __init__(self, save_folder, opt, argv):
        self.save_folder = save_folder
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)

        if os.path.isfile(os.path.join(save_folder, 'logfile.log')):
            os.remove(os.path.join(save_folder, 'logfile.log'))

        file_log_handler = logging.FileHandler(os.path.join(save_folder, 'logfile.log'))
        self.logger.addHandler(file_log_handler)

        stdout_log_handler = logging.StreamHandler(sys.stdout)
        self.logger.addHandler(stdout_log_handler)
        # commend line
        self.logger.info("# COMMEND LINE ===========")
        self.logger.info(argv)
        self.logger.info("# =====================")
        # meta info
        self.logger.info("# META INFO ===========")
        attrs = vars(opt)
        for item in attrs.items():
            self.logger.info("%s: %s"%item)
        # self.logger.info("Saved in: {}".format(save_folder))
        self.logger.info("# =====================")

    def log_value(self, epoch, *info_pairs):
        log_str = "Epoch: {}; ".format(epoch)
        for name, value in info_pairs:
            log_str += (str(name) + ": {}; ").format(value)
        self.logger.info(log_str)

    def save_value(self, name, list_of_values):
        np.save(os.path.join(self.save_folder, name), list_of_values)

