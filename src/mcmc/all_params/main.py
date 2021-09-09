"""
{This script}
"""
__author__ = '{Mehnaaz Asad}'

from settings import Settings
from preprocess import Preprocess
from analysis import Analysis

def main():
    Settings.Initalize_Global_Settings()
    Preprocess.Load_Into_Dataframes()
    Analysis.Core()


if __name__ == '__main__':
    main()