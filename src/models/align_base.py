import torch
from transformers import AlignProcessor, AlignModel
from transformers.models.align.modeling_align import ALIGN_START_DOCSTRING, ALIGN_INPUTS_DOCSTRING, AlignOutput
from transformers.models.align.configuration_align import AlignConfig
from typing import Any, Optional, Tuple, Union
from transformers.utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)

@add_start_docstrings(ALIGN_START_DOCSTRING)
class MyAlignModel(AlignModel):
    def __init__(self, config: AlignConfig):
        super().__init__(config)
    # @add_start_docstrings_to_model_forward(ALIGN_INPUTS_DOCSTRING)
    # @replace_return_docstrings(output_type=AlignOutput, config_class=AlignConfig)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        return_loss: Optional[bool] = True,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, AlignOutput]:
        return super().forward(
            input_ids= input_ids,
            pixel_values= pixel_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            return_loss=return_loss, # Enable the model to calculate and return loss
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

class align_base:
    '''
    Model card: https://huggingface.co/kakaobrain/align-base
    Usage examples: https://huggingface.co/docs/transformers/en/model_doc/align
    '''
    def __init__(self) -> None:
        self.processor = AlignProcessor.from_pretrained("kakaobrain/align-base")
        # self.model = AlignModel.from_pretrained("kakaobrain/align-base")
        self.model = MyAlignModel.from_pretrained("kakaobrain/align-base")
        self.device = torch.device('cuda' if torch.cuda.is_available() else  'cpu')
        self.model.to(self.device)

    def forward(self, texts: list, images: list):
        inputs = self.processor(text=texts, images=images, return_tensors="pt")
        inputs.to(self.device)
        with torch.no_grad():
            # for i in range(10000):
            outputs = self.model(**inputs) # It only takes 2GB of GPU memory for inference

        # this is the image-text similarity score
        print(outputs.keys())
        logits_per_image = outputs.logits_per_image

        # we can take the softmax to get the label probabilities
        probs = logits_per_image.softmax(dim=1)
        return probs