"""
{This script}
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

    analysis.Core(experiments)
    experiments.Run_Experiments()
    
    models = analysis.Mocks_And_Models(experiments)
    # Plotting.Plot_Core()
    # Plotting.Plot_Experiments()


if __name__ == '__main__':
    main()