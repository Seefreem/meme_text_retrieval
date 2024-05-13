import torch

def similarity_align(inputs, align_model, device):
    '''
    Calculate the similarity matrix of images and texts.
    
    Inputs:
        inputs [transformers.tokenization_utils_base.BatchEncoding]: tokenized inputs.
        align_model [align_base]: model
        device: device
    Output:
        text2image retrieval similarities
    '''
    inputs.to(device)
    with torch.no_grad():
        text_embeds = align_model.model.get_text_features(input_ids=inputs['input_ids'],
                                                        token_type_ids=inputs['token_type_ids'],
                                                        attention_mask=inputs['attention_mask'])
        # normalized features
        text_embeds = (text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)).cpu()
        print(text_embeds.shape) # torch.Size([558, 640])
        
        # Calculate the image embeddings one by one, due to the limited GPU memory.
        image_embeds = []
        for i in range(len(inputs['pixel_values'])): 
            image_embedding = align_model.model.get_image_features(pixel_values=inputs['pixel_values'][i].unsqueeze(0))
            # normalized features
            image_embeds.append(image_embedding.cpu())
            
        # Convert list of tensors to a single tensor
        image_embeds = torch.stack(image_embeds).squeeze(1)
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        print(image_embeds.shape)

    # cosine similarity as logits
    # text2image retrieval similarities
    logits_per_text = torch.matmul(text_embeds, image_embeds.t()) / align_model.model.temperature.cpu().detach().numpy()
    return logits_per_text


def similarity_clip(inputs, model, device):
    '''
    Calculate the similarity matrix of images and texts.
    
    Inputs:
        inputs [transformers.tokenization_utils_base.BatchEncoding]: tokenized inputs.
        model [align_base]: model
        device: device
    Output:
        text2image retrieval similarities
    '''
    inputs.to(device)
    with torch.no_grad():
        text_embeds = model.model.get_text_features(input_ids=inputs['input_ids'],
                                                        attention_mask=inputs['attention_mask'])
        # normalized features
        text_embeds = (text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)).cpu()
        print(text_embeds.shape) 
        
        # Calculate the image embeddings one by one, due to the limited GPU memory.
        image_embeds = []
        for i in range(len(inputs['pixel_values'])): 
            image_embedding = model.model.get_image_features(pixel_values=inputs['pixel_values'][i].unsqueeze(0))
            # normalized features
            image_embeds.append(image_embedding.cpu())
            
        # Convert list of tensors to a single tensor
        image_embeds = torch.stack(image_embeds).squeeze(1)
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        print(image_embeds.shape)

    # cosine similarity as logits
    # text2image retrieval similarities
    logit_scale = model.model.logit_scale.exp().cpu().detach().numpy()
    logits_per_text = torch.matmul(text_embeds, image_embeds.t())  * logit_scale
    return logits_per_text
