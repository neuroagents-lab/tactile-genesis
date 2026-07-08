"""Robot configuration option presets."""

from eden.options.robots.aloha_vx300s import Aloha
from eden.options.robots.franka_emika_panda import FrankaEmikaPanda, FrankaResearch3
from eden.options.robots.franka_hand import FrankaHand
from eden.options.robots.i2rt_yam import Yam, YAM_ACTION_SCALE
from eden.options.robots.inspire_rh56 import InspireRH56_L, InspireRH56_R
from eden.options.robots.leap_hand import LeapHand_L, LeapHand_R
from eden.options.robots.mano_hand import ManoHand_L, ManoHand_R
from eden.options.robots.robotera_xhand import XHand1_L, XHand1_R
from eden.options.robots.robotiq_2f85 import Robotiq2f85
from eden.options.robots.sharpa_wave import SharpaWave_L, SharpaWave_R
from eden.options.robots.tesollo_dg5f import TesolloDG5F_L, TesolloDG5F_R
from eden.options.robots.trossen_widowx250 import WidowX250
from eden.options.robots.universal_ur5e import UR5e
from eden.options.robots.universal_ur10e import UR10e


__all__ = [
    "Aloha",
    "FrankaEmikaPanda",
    "FrankaResearch3",
    "FrankaHand",
    "InspireRH56_L",
    "InspireRH56_R",
    "LeapHand_L",
    "LeapHand_R",
    "ManoHand_L",
    "ManoHand_R",
    "Robotiq2f85",
    "SharpaWave_L",
    "SharpaWave_R",
    "TesolloDG5F_L",
    "TesolloDG5F_R",
    "UR5e",
    "UR10e",
    "WidowX250",
    "XHand1_L",
    "XHand1_R",
    "Yam",
    "YAM_ACTION_SCALE",
]
