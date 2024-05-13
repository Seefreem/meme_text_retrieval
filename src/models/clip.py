import torch
from transformers import CLIPProcessor, CLIPModel

class clip:
    '''
    Model card: https://huggingface.co/openai/clip-vit-base-patch32
    '''
    def __init__(self) -> None:
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

        self.device = torch.device('cuda' if torch.cuda.is_available() else  'cpu')
        self.model.to(self.device)

    def forward(self, texts: list, images: list):
        inputs = self.processor(text=texts, images=images, return_tensors="pt")
        inputs.to(self.device)
        with torch.no_grad():
            # for i in range(10000):
            outputs = self.model(**inputs) # It only takes 2GB of GPU memory for inference

        # this is the image-text similarity score
        logits_per_image = outputs.logits_per_image

        # we can take the softmax to get the label probabilities
        probs = logits_per_image.softmax(dim=1)
        return probs