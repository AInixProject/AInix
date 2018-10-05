from ainix_common.parsing.parseast import AstObjectChoiceSet, ObjectChoiceNode
from ainix_kernel.models.EncoderDecoder.encoders import QueryEncoder
from ainix_kernel.models.model_types import StringTypeTranslateCF

class EncDecModel(StringTypeTranslateCF):
    def __init__(self, query_encoder: QueryEncoder, tree_decoder):
        self.query_encoder = query_encoder
        pass

    def predict(
        self,
        x_string: str,
        y_type_name: str,
        use_only_train_data: bool
    ) -> ObjectChoiceNode:
        pass

    def train(
        self,
        x_string: str,
        y_ast: AstObjectChoiceSet,
        teacher_force_path: ObjectChoiceNode
    ):
        pass