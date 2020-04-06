
class Statistics():
    def __init__(self):
        self.solved_after = 0
        self.frames_total = 0
        self.solved = False
        self.steps_total = []
        self.score_total = []
        self.episode = 0
        self.egreedy = 0

    def reset(self, hard):
        if hard:
            self.solved_after = 0
            self.frames_total = 0
            self.episode = 0
            self.egreedy = 0
            self.solved = False
            self.steps_total = []
            self.score_total = []
        else:
            self.steps_total = []
            self.score_total = []

    def add(self, stat_name, value, continous):
        if continous:
            getattr(self,stat_name).append(value)
        else:
            setattr(self,stat_name, getattr(self,stat_name) + value)

    def mean(self,stat_name, part_mean = False, len_o_subset = None):
        if part_mean:
            assert len_o_subset is not None, 'len_o_subset cant be None when part_mean is true'
            return sum(getattr(self,stat_name)[-len_o_subset:])/len_o_subset
        else:
            return sum(getattr(self,stat_name))/len(getattr(self,stat_name))

    def set(self, stat_name ,value):
        setattr(self,stat_name,value)

    def get(self,stat_name):
        return getattr(self,stat_name)

    def string_report(self, report_interval):
        print('\n********* Episode %i **************\
                          \navg score last %i: %.2f, avg score last 100: %.2f, avg score total: %.2f\
                          \negreedy: %.2f frames total: %i, score:  %.2f'

                          % (
                              self.episode,
                              report_interval,
                              self.mean('score_total',True,report_interval),
                              self.mean('score_total', True, 100),
                              self.mean('score_total'),
                              self.egreedy,
                              self.frames_total,
                              self.score_total[-1]
                          )
            )