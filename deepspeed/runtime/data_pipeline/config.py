'''
Copyright 2022 The Microsoft DeepSpeed Team
'''
from pydantic import Field, root_validator
from enum import Enum
from .constants import *
import copy
from ..config_utils import DeepSpeedConfigModel


def get_data_efficiency_config(param_dict):
    return DeepSpeedDataEfficiencyConfig(**param_dict.get("data_efficiency", {}))


class ClusteringTypeEnum(str, Enum):
    schedule_base = "schedule_based"
    single_cluster = "single_cluster"

class DifficultyTypeEnum(str, Enum):
    value = "value"
    percentile = "percentile"

class CurriculumScheduleTypeEnum(str, Enum):
    fixed_linear = "fixed_linear"
    fixed_root = "fixed_root"
    fixed_discrete = "fixed_discrete"
    custom = "custom"


class CurriculumScheduleConfig(DeepSpeedConfigModel):
    total_curriculum_step: int = None
    difficulty_step: int = None
    root_degree: int = None
    difficulty: List[int] = None
    max_step: List[int] = None


class CurriculumMetricsConfig(DeepSpeedConfigModel):
    index_to_sample_path: str = None
    index_to_metric_path: str = None
    difficulty_type: DifficultTypeEnum = None
    clustering_type: ClusterTypeEnum = None
    min_difficulty: int = None
    max_difficulty: int = None
    schedule_type: CurriculumScheduleTypeEnum = None
    schedule_config: ScheduleConfig = {}


class CurriculumLearningConfig(DeepSpeedConfigModel):
    enabled: bool = False
    data_cluser_path: str = None
    curriculum_metrics: Dict[str,CurriculumMetricsConfig] = {}

    @root_validator
    def assert_curriculum_metrics(cls, values):
        if values.get("enabled"):
            assert values[curriculum_metrics] != {}, "Curriculum learning is enabled, 'curriculum_metrics' must be specified"
        return values


class DataSamplingConfig(DeepSpeedConfigModel):
    enabled: bool = False
    num_epochs: int = Field(1000, ge=0)
    num_workers: int = Field(0, ge=0)
    curriculum_learning: CurriculumLearningConfig = {}


class ModelTypeEnum(str, Enum):
    encoder = "encoder"
    decoder = "decoder"


class HiddenStateOrderEnum(str, Enum):
    batch_seq_dim = "batch_seq_dim"
    seq_batch_dim = "seq_batch_dim"


class LTDScheduleTypeEnum(str, Enum):
    fixed_linear = "fixed_linear"


class LTDScheduleConfig(DeepSpeedConfigModel):
    require_steps: int = None
    seq_per_step: int = None


class RandomLTDScheduleConfig(DeepSpeedConfigModel):
    min_value: int = None
    max_value: int = None
    schedule_type: LTDScheduleTypeEnum = None
    schedule_config: LTDScheduleConfig = {}


class RandomLTDConfig(DeepSpeedConfigModel):
    enabled: bool = False
    total_layer_num: int = Field(None, ge=0)
    random_ltd_layer_num: int = Field(None, ge=0)
    random_ltd_layer_id: List[int]
    model_mask_name: str = None
    model_type: ModelTypeEnum = None
    hidden_state_order: HiddenStateOrderEnum = None
    random_ltd_schedule: RandomLTDScheduleConfig = {}

    @root_validator
    def check_random_ltd_layer_id(cls, values):
        layer_num = values.get("random_ltd_layer_num")
        layer_id_list = values.get("random_ltd_layer_id")
        assert len(layer_id_list) == layer_num, "'random_ltd_layer_id' list length must match value given in 'random_ltd_layer_num'"
        return values


class DataRoutingConfig(DeepSpeedConfigModel):
    enabled: bool = False
    random_ltd: RandomLTDConfig = {}


class DeepSpeedDataEfficiencyConfig(DeepSpeedConfigModel):
    enabled: bool = False
    seed: int = Field(1234, ge=0)
    data_sampling: DataSamplingConfig = {}
    data_routing: DataRoutingConfig = {}


def get_curriculum_enabled_legacy(param_dict):
    if CURRICULUM_LEARNING_LEGACY in param_dict.keys():
        return get_scalar_param(param_dict[CURRICULUM_LEARNING_LEGACY],
                                CURRICULUM_ENABLED_LEGACY,
                                CURRICULUM_ENABLED_DEFAULT_LEGACY)
    else:
        return False


def get_curriculum_params_legacy(param_dict):
    if CURRICULUM_LEARNING_LEGACY in param_dict.keys():
        curriculum_params = copy.copy(param_dict[CURRICULUM_LEARNING_LEGACY])
        curriculum_params.pop(CURRICULUM_ENABLED_LEGACY)
        return curriculum_params
    else:
        return False
