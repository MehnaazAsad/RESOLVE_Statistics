"""
{This script is the entry point to making plots of final results for 
Asad et al. 2023, in prep.}
"""
__author__ = '{Mehnaaz Asad}'

from settings import Settings
from preprocess import Preprocess
from analysis import Analysis
from experiments import Experiments
from plotting import Plotting

def main():    
    settings = Settings()
    settings.Initalize_Global_Settings()

    preprocess = Preprocess(settings)
    preprocess.Load_Into_Dataframes()
    
    analysis = Analysis(preprocess)
    experiments = Experiments(analysis)

    data = analysis.Core(experiments)
    data_experimentals = experiments.Run_Experiments()
    
    models, best_fit, gals_df = analysis.Mocks_And_Models(experiments)

    plotting = Plotting(preprocess)
    plotting.Plot_Core(data, models, best_fit)
    plotting.Plot_Experiments(data, data_experimentals, models, best_fit)


if __name__ == '__main__':
    main()