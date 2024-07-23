import os
import sys

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from envs.BetaAviary import BetaAviary
from envs.CtrlAviary import CtrlAviary
from envs.HoverAviary import HoverAviary
from envs.MultiHoverAviary import MultiHoverAviary
from envs.VelocityAviary import VelocityAviary
