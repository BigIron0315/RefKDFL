from .bottleneck_blocks import Bottle_2block_14_32, Bottle_2block_20_27, Bottle_2block_32_21, Bottle_2block_56_16, Bottle_2block_110_11
from .bottleneck_blocks import Bottle_2block_14_42, Bottle_2block_20_35, Bottle_2block_32_28, Bottle_2block_56_21, Bottle_2block_110_15
from .bottleneck_blocks import Bottle_2block_14_60, Bottle_2block_20_51, Bottle_2block_32_39, Bottle_2block_56_29, Bottle_2block_110_21
from .bottleneck_blocks import Bottle_3block_20_32, Bottle_3block_29_27, Bottle_3block_47_21, Bottle_3block_83_16, Bottle_3block_174_11
from .bottleneck_blocks import Bottle_3block_20_42, Bottle_3block_29_35, Bottle_3block_47_28, Bottle_3block_83_21, Bottle_3block_174_15
from .bottleneck_blocks import Bottle_3block_20_60, Bottle_3block_29_51, Bottle_3block_47_39, Bottle_3block_83_29, Bottle_3block_174_21
from .bottleneck_blocks import Bottle_4block_26_32, Bottle_4block_38_27, Bottle_4block_62_21, Bottle_4block_110_16, Bottle_4block_218_11
from .bottleneck_blocks import Bottle_4block_26_42, Bottle_4block_38_35, Bottle_4block_62_28, Bottle_4block_110_21, Bottle_4block_218_15
from .bottleneck_blocks import Bottle_4block_26_60, Bottle_4block_38_51, Bottle_4block_62_39, Bottle_4block_110_29, Bottle_4block_218_21

from .basic_blocks import Basic_3block_2_16, Basic_3block_3_12, Basic_3block_4_10, Basic_3block_6_8, Basic_3block_8_7
from .basic_blocks import Basic_3block_2_24, Basic_3block_3_19, Basic_3block_4_16, Basic_3block_6_13, Basic_3block_8_11
from .basic_blocks import Basic_3block_2_48, Basic_3block_3_37, Basic_3block_4_31, Basic_3block_6_25, Basic_3block_8_21
from .basic_blocks import Basic_4block_1_28, Basic_4block_2_16, Basic_4block_3_12, Basic_4block_4_10, Basic_4block_6_8, Basic_4block_8_7
from .basic_blocks import Basic_4block_1_42, Basic_4block_2_24, Basic_4block_3_19, Basic_4block_4_16, Basic_4block_6_13, Basic_4block_8_11
from .basic_blocks import Basic_4block_1_83, Basic_4block_2_48, Basic_4block_3_37, Basic_4block_4_31, Basic_4block_6_25, Basic_4block_8_21
from .basic_blocks import Basic_5block_2_16, Basic_5block_3_12, Basic_5block_4_10, Basic_5block_6_8, Basic_5block_8_7
from .basic_blocks import Basic_5block_2_24, Basic_5block_3_19, Basic_5block_4_16, Basic_5block_6_13, Basic_5block_8_11
from .basic_blocks import Basic_5block_2_48, Basic_5block_3_37, Basic_5block_4_31, Basic_5block_6_25, Basic_5block_8_21

from .residual_blocks import Residual_2block_10_33, Residual_2block_14_27, Residual_2block_22_21, Residual_2block_38_16, Residual_2block_74_11
from .residual_blocks import Residual_2block_10_44, Residual_2block_14_36, Residual_2block_22_28, Residual_2block_38_21, Residual_2block_74_15
from .residual_blocks import Residual_2block_10_62, Residual_2block_14_51, Residual_2block_22_39, Residual_2block_38_29, Residual_2block_74_21
from .residual_blocks import Residual_3block_14_33, Residual_3block_20_27, Residual_3block_32_21, Residual_3block_56_16, Residual_3block_110_11
from .residual_blocks import Residual_3block_14_44, Residual_3block_20_36, Residual_3block_32_28, Residual_3block_56_21, Residual_3block_110_15
from .residual_blocks import Residual_3block_14_62, Residual_3block_20_51, Residual_3block_32_39, Residual_3block_56_29, Residual_3block_110_21
from .residual_blocks import Residual_4block_18_33, Residual_4block_26_27, Residual_4block_42_21, Residual_4block_74_16, Residual_4block_146_11
from .residual_blocks import Residual_4block_18_44, Residual_4block_26_36, Residual_4block_42_28, Residual_4block_74_21, Residual_4block_146_15
from .residual_blocks import Residual_4block_18_62, Residual_4block_26_51, Residual_4block_42_39, Residual_4block_74_29, Residual_4block_146_21


model_dict = {
    'Bottle_2block_14_32' : Bottle_2block_14_32, 
    'Bottle_2block_20_27' : Bottle_2block_20_27, 
    'Bottle_2block_32_21' : Bottle_2block_32_21, 
    'Bottle_2block_56_16' : Bottle_2block_56_16, 
    'Bottle_2block_110_11' : Bottle_2block_110_11,
    'Bottle_2block_14_42' : Bottle_2block_14_42, 
    'Bottle_2block_20_35' : Bottle_2block_20_35, 
    'Bottle_2block_32_28' : Bottle_2block_32_28, 
    'Bottle_2block_56_21' : Bottle_2block_56_21, 
    'Bottle_2block_110_15' : Bottle_2block_110_15,
    'Bottle_2block_14_60' : Bottle_2block_14_60, 
    'Bottle_2block_20_51' : Bottle_2block_20_51, 
    'Bottle_2block_32_39' : Bottle_2block_32_39, 
    'Bottle_2block_56_29' : Bottle_2block_56_29, 
    'Bottle_2block_110_21' : Bottle_2block_110_21,
    'Bottle_3block_20_32' : Bottle_3block_20_32, 
    'Bottle_3block_29_27' : Bottle_3block_29_27, 
    'Bottle_3block_47_21' : Bottle_3block_47_21, 
    'Bottle_3block_83_16' : Bottle_3block_83_16, 
    'Bottle_3block_174_11' : Bottle_3block_174_11,
    'Bottle_3block_20_42' : Bottle_3block_20_42, 
    'Bottle_3block_29_35' : Bottle_3block_29_35, 
    'Bottle_3block_47_28' : Bottle_3block_47_28, 
    'Bottle_3block_83_21' : Bottle_3block_83_21, 
    'Bottle_3block_174_15' : Bottle_3block_174_15,
    'Bottle_3block_20_60' : Bottle_3block_20_60, 
    'Bottle_3block_29_51' : Bottle_3block_29_51, 
    'Bottle_3block_47_39' : Bottle_3block_47_39, 
    'Bottle_3block_83_29' : Bottle_3block_83_29, 
    'Bottle_3block_174_21' : Bottle_3block_174_21,
    'Bottle_4block_26_32' : Bottle_4block_26_32, 
    'Bottle_4block_38_27' : Bottle_4block_38_27, 
    'Bottle_4block_62_21' : Bottle_4block_62_21, 
    'Bottle_4block_110_16' : Bottle_4block_110_16, 
    'Bottle_4block_218_11' : Bottle_4block_218_11,
    'Bottle_4block_26_42' : Bottle_4block_26_42, 
    'Bottle_4block_38_35' : Bottle_4block_38_35, 
    'Bottle_4block_62_28' : Bottle_4block_62_28, 
    'Bottle_4block_110_21' : Bottle_4block_110_21, 
    'Bottle_4block_218_15' : Bottle_4block_218_15,
    'Bottle_4block_26_60' : Bottle_4block_26_60, 
    'Bottle_4block_38_51' : Bottle_4block_38_51, 
    'Bottle_4block_62_39' : Bottle_4block_62_39, 
    'Bottle_4block_110_29' : Bottle_4block_110_29, 
    'Bottle_4block_218_21' : Bottle_4block_218_21,

    'Basic_3block_2_16' : Basic_3block_2_16, 
    'Basic_3block_3_12' : Basic_3block_3_12, 
    'Basic_3block_4_10' : Basic_3block_4_10, 
    'Basic_3block_6_8' : Basic_3block_6_8, 
    'Basic_3block_8_7' : Basic_3block_8_7,
    'Basic_3block_2_24' : Basic_3block_2_24, 
    'Basic_3block_3_19' : Basic_3block_3_19, 
    'Basic_3block_4_16' : Basic_3block_4_16, 
    'Basic_3block_6_13' : Basic_3block_6_13, 
    'Basic_3block_8_11' : Basic_3block_8_11,
    'Basic_3block_2_48' : Basic_3block_2_48, 
    'Basic_3block_3_37' : Basic_3block_3_37, 
    'Basic_3block_4_31' : Basic_3block_4_31, 
    'Basic_3block_6_25' : Basic_3block_6_25, 
    'Basic_3block_8_21' : Basic_3block_8_21,
    'Basic_4block_1_28' : Basic_4block_1_28,
    'Basic_4block_2_16' : Basic_4block_2_16, 
    'Basic_4block_3_12' : Basic_4block_3_12, 
    'Basic_4block_4_10' : Basic_4block_4_10, 
    'Basic_4block_6_8' : Basic_4block_6_8, 
    'Basic_4block_8_7' : Basic_4block_8_7,
    'Basic_4block_1_42' : Basic_4block_1_42,
    'Basic_4block_2_24' : Basic_4block_2_24, 
    'Basic_4block_3_19' : Basic_4block_3_19, 
    'Basic_4block_4_16': Basic_4block_4_16, 
    'Basic_4block_6_13' : Basic_4block_6_13, 
    'Basic_4block_8_11': Basic_4block_8_11,
    'Basic_4block_1_83' : Basic_4block_1_83,
    'Basic_4block_2_48' : Basic_4block_2_48,
    'Basic_4block_3_37' : Basic_4block_3_37, 
    'Basic_4block_4_31' : Basic_4block_4_31, 
    'Basic_4block_6_25' : Basic_4block_6_25, 
    'Basic_4block_8_21' : Basic_4block_8_21,
    'Basic_5block_2_16' : Basic_5block_2_16, 
    'Basic_5block_3_12' : Basic_5block_3_12, 
    'Basic_5block_4_10' : Basic_5block_4_10, 
    'Basic_5block_6_8' : Basic_5block_6_8, 
    'Basic_5block_8_7' : Basic_5block_8_7,
    'Basic_5block_2_24' : Basic_5block_2_24, 
    'Basic_5block_3_19' : Basic_5block_3_19, 
    'Basic_5block_4_16' : Basic_5block_4_16, 
    'Basic_5block_6_13' : Basic_5block_6_13, 
    'Basic_5block_8_11' : Basic_5block_8_11,
    'Basic_5block_2_48' : Basic_5block_2_48, 
    'Basic_5block_3_37' : Basic_5block_3_37, 
    'Basic_5block_4_31' : Basic_5block_4_31, 
    'Basic_5block_6_25' : Basic_5block_6_25, 
    'Basic_5block_8_21' : Basic_5block_8_21,

    'Residual_2block_10_33' : Residual_2block_10_33, 
    'Residual_2block_14_27' : Residual_2block_14_27, 
    'Residual_2block_22_21' : Residual_2block_22_21, 
    'Residual_2block_38_16' : Residual_2block_38_16, 
    'Residual_2block_74_11' : Residual_2block_74_11,
    'Residual_2block_10_44' : Residual_2block_10_44, 
    'Residual_2block_14_36' : Residual_2block_14_36, 
    'Residual_2block_22_28' : Residual_2block_22_28, 
    'Residual_2block_38_21' : Residual_2block_38_21, 
    'Residual_2block_74_15' : Residual_2block_74_15,
    'Residual_2block_10_62' : Residual_2block_10_62, 
    'Residual_2block_14_51' : Residual_2block_14_51, 
    'Residual_2block_22_39' : Residual_2block_22_39, 
    'Residual_2block_38_29' : Residual_2block_38_29, 
    'Residual_2block_74_21' : Residual_2block_74_21,    
    'Residual_3block_14_33' : Residual_3block_14_33,
    'Residual_3block_20_27' : Residual_3block_20_27,
    'Residual_3block_32_21' : Residual_3block_32_21,
    'Residual_3block_56_16' : Residual_3block_56_16,
    'Residual_3block_110_11' : Residual_3block_110_11,
    'Residual_3block_14_44' : Residual_3block_14_44,
    'Residual_3block_20_36' : Residual_3block_20_36,
    'Residual_3block_32_28' : Residual_3block_32_28,
    'Residual_3block_56_21' : Residual_3block_56_21,
    'Residual_3block_110_15' : Residual_3block_110_15,
    'Residual_3block_14_62' : Residual_3block_14_62, 
    'Residual_3block_20_51' : Residual_3block_20_51, 
    'Residual_3block_32_39' : Residual_3block_32_39, 
    'Residual_3block_56_29' : Residual_3block_56_29, 
    'Residual_3block_110_21' : Residual_3block_110_21,
    'Residual_4block_18_33' : Residual_4block_18_33, 
    'Residual_4block_26_27' : Residual_4block_26_27, 
    'Residual_4block_42_21' : Residual_4block_42_21, 
    'Residual_4block_74_16' : Residual_4block_74_16, 
    'Residual_4block_146_11' : Residual_4block_146_11,
    'Residual_4block_18_44' : Residual_4block_18_44, 
    'Residual_4block_26_36' : Residual_4block_26_36, 
    'Residual_4block_42_28' : Residual_4block_42_28, 
    'Residual_4block_74_21' : Residual_4block_74_21, 
    'Residual_4block_146_15' : Residual_4block_146_15,
    'Residual_4block_18_62' : Residual_4block_18_62, 
    'Residual_4block_26_51' : Residual_4block_26_51, 
    'Residual_4block_42_39' : Residual_4block_42_39, 
    'Residual_4block_74_29' : Residual_4block_74_29, 
    'Residual_4block_146_21' : Residual_4block_146_21,

}