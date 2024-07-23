
from typing import List, Tuple
import matplotlib.pyplot as plt
from planarEnvVEC import environment_agent


import sys
import os

# if 'CUDA_PATH' in os.environ:
    # from env_gpu import environment_agent_gpu

import argparse


def main(boundaries: tuple = (0,0),
         N         : int = 10,
         N_preys   : int = 10,
         no_sensor : float = 0.0,
         experiment_id: str = "1",
         steps     : int = 10000,
         save      : bool = False,
         sensor_range: int = 5,
         sensing_range: int = 2,
         draw      : bool = False,
         no_progress_bar: bool = False,
         error_range: float = 0.05,
        #  min_distance: float = 5,
         blind_spots: bool = False,
         draw_circles: bool = False,
         gpu: bool = False,
         pdm: bool = False,
         kappa: float = 3.0,
         Dp: float = 3.0,
         Dp_pm: float = 3.0,
            **kwargs) -> None:


    # Run the simulation
    if pdm:
        filename      = f'Agents[{N}]_Preys[{N_preys}]_steps[{steps}]_precentageNoSensor[{no_sensor}]_sensingRange[{sensing_range}]_sensorRange[{sensor_range}]_pdm[{pdm}]_exp[{experiment_id}]'
    else:
        filename      = f'Agents[{N}]_Preys[{N_preys}]_steps[{steps}]_precentageNoSensor[{no_sensor}]_sensingRange[{sensing_range}]_sensorRange[{sensor_range}]_pdm[{pdm}]_exp[{experiment_id}]'


    if gpu:
        assert draw == False and 'environment_agent_gpu' in globals(), "Cannot draw the simulation with GPU or environment_agent_gpu is not defined!!"
        # env = environment_agent_gpu(boundaries, 
        #                             N, 
        #                             N_preys, 
        #                             no_sensor, 
        #                             experiment_id, 
        #                             steps, 
        #                             save, 
        #                             error_range,
        #                             sensor_range, 
        #                             sensing_range, 
        #                             percentage_no_sensor=no_sensor, 
        #                             draw=draw, 
        #                             no_progress_bar=no_progress_bar,
        #                             filename=filename,
        #                             min_distance=min_distance,
        #                             blind_spots=blind_spots,
        #                             draw_circles=draw_circles,
        #                             pdm=pdm,
        #                             **kwargs)
        print("GPU implemented but too slow to run")
    else:
        env = environment_agent(boundaries, 
                                N, 
                                N_preys, 
                                no_sensor, 
                                experiment_id, 
                                steps, 
                                save, 
                                error_range,
                                sensor_range, 
                                sensing_range, 
                                percentage_no_sensor=no_sensor, 
                                draw=draw, 
                                no_progress_bar=no_progress_bar,
                                filename=filename,
                                # min_distance=min_distance,
                                blind_spots=blind_spots,
                                draw_circles=draw_circles,
                                pdm=pdm,
                                kappa=kappa,
                                Dp=Dp,
                                Dp_pm=Dp_pm,
                                **kwargs)
    env.run()



if __name__ == "__main__":
    # Define the parameters for the experiment
    parser = argparse.ArgumentParser(description='Run the simulation.')
    parser.add_argument('--boundaries',      type=int,   default=[0,0], nargs=2,  help='Boundaries of the environment'               )
    parser.add_argument('--N',               type=int,   default=10,              help='Number of predators'                         )
    parser.add_argument('--N_preys',         type=int,   default=10,              help='Number of preys'                             )
    parser.add_argument('--no_sensor',       type=float, default=0.0,             help='Percentage of predators without sensor'      )
    parser.add_argument('--experiment_id',   type=str,   default="1",             help='Experiment ID'                               )
    parser.add_argument('--steps',           type=int,   default=50000,           help='Number of steps'                             )
    parser.add_argument('--save',            action='store_true',                 help='Save the simulation'                         )
    parser.add_argument('--sensor_range',    type=int,   default=3,               help='Sensor range predator'                       )
    parser.add_argument('--sensing_range',   type=int,   default=3,               help='Sensing range prey'                          )
    parser.add_argument('--draw',            action='store_true',                 help='Draw the simulation'                         )
    parser.add_argument('--no_progress_bar', action='store_true',                 help='No progress bar'                             )
    parser.add_argument('--error_range',     type=float,  default=0.05,           help='Error range'                                 )
    # parser.add_argument('--min_distance',    type=float,  default=5.0,            help='Minimum distance between predators and preys')
    parser.add_argument('--blind_spots',     action='store_true',                 help='Blind spots'                                 )
    parser.add_argument('--draw_circles',    action='store_true',                 help='Draw circles'                                )
    parser.add_argument('--gpu',             action='store_true',                 help='Use GPU'                                     )
    parser.add_argument('--pdm',             action='store_true',                 help='Use PDM'                                     )
    parser.add_argument('--folder',          type=str,   default='results',       help='Folder to save the results'                  )
    parser.add_argument('--save_last_step',  action='store_true',                 help='Save the last step'                          )
    parser.add_argument('--plot_vel',        action='store_true',                 help='Plot velocity'                               )
    #############################################################################
    # CONSTANTS
    parser.add_argument('--sigma_init',      type=float, default=1.5,             help='Initial value of sigma'                      )
    parser.add_argument('--alpha',           type=float, default=1.0,             help='Alpha'                                       )
    parser.add_argument('--beta',            type=float, default=0.0,             help='Beta'                                        )
    parser.add_argument('--gamma',           type=float, default=0.0,             help='Gamma'                                       )
    parser.add_argument('--kappa',           type=float, default=1.0,             help='Kappa'                                       )
    parser.add_argument('--alpha_prey',      type=float, default=1.0,             help='Alpha prey'                                  )
    parser.add_argument('--Dp',              type=float, default=4.0,             help='Dp'                                          )
    parser.add_argument('--sigma_i_predator',type=float, default=0.7,             help='Sigma_i_predator'                            )
    parser.add_argument('--sigma_i_pred_non_sensing', type=float, default=0.75,   help='Sigma_i_pred_non_sensing'                    )
    parser.add_argument('--sigma_i_pred_DM',           type=float, default=1.5,   help='Sigma_i_pred_DM'                             )
    parser.add_argument('--sigma_i_pred_non_sensing_DM', type=float, default=2.0, help='Sigma_i_pred_non_sensing_DM'                 )
    parser.add_argument('--Dp_prey',         type=float, default=3.5,             help='Dp_prey'                                     )
    parser.add_argument('--Dp_pm',           type=float, default=3.5,             help='Dp_pm'                                       )
    parser.add_argument('--sigma_i_prey',    type=float, default=0.7,             help='Sigma_i_prey'                                )
    #############################################################################
    args = parser.parse_args()

    kwargs = vars(args)
    main(**kwargs)

    # if 'CUDA_PATH' in os.environ:
