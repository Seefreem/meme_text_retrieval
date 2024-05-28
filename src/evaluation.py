import torch
import time
from collections import defaultdict, deque
import datetime
import numpy as np
import torch.nn.functional as F

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
        text_embeds = align_model.get_text_features(input_ids=inputs['input_ids'],
                                                        token_type_ids=inputs['token_type_ids'],
                                                        attention_mask=inputs['attention_mask'])
        # normalized features
        text_embeds = (text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)).cpu()
        print(text_embeds.shape) # torch.Size([558, 640])
        
        # Calculate the image embeddings one by one, due to the limited GPU memory.
        image_embeds = []
        for i in range(len(inputs['pixel_values'])): 
            image_embedding = align_model.get_image_features(pixel_values=inputs['pixel_values'][i].unsqueeze(0))
            # normalized features
            image_embeds.append(image_embedding.cpu())
            
        # Convert list of tensors to a single tensor
        image_embeds = torch.stack(image_embeds).squeeze(1)
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        print(image_embeds.shape)

    # cosine similarity as logits
    # text2image retrieval similarities
    logits_per_text = torch.matmul(text_embeds, image_embeds.t()) / align_model.temperature.cpu().detach().numpy()
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

@torch.no_grad()
def evaluation(model, data_loader, tokenizer, device, config):
    # test
    model.eval() 
    print('Computing features for evaluation...')
    start_time = time.time()  

    texts = data_loader.dataset.text   
    num_text = len(texts)
    text_bs = 256
    text_feats = []
    text_embeds = []  
    text_atts = []
    # Compute text features, embeddings and attentions masks.
    for i in range(0, num_text, text_bs):
        text = texts[i: min(num_text, i+text_bs)]   
        text_input = tokenizer(text, padding='max_length', truncation=True, max_length=30, return_tensors="pt").to(device) 
        text_output = model.text_encoder(text_input.input_ids, attention_mask = text_input.attention_mask, mode='text')  
        text_feat = text_output.last_hidden_state
        text_embed = F.normalize(model.text_proj(text_feat[:,0,:]))
        text_embeds.append(text_embed)   
        text_feats.append(text_feat)
        text_atts.append(text_input.attention_mask)
    text_embeds = torch.cat(text_embeds,dim=0)
    text_feats = torch.cat(text_feats,dim=0)
    text_atts = torch.cat(text_atts,dim=0)

    image_feats = []
    image_embeds = []
    # Compute image features, embeddings and attentions masks.
    for image, img_id in data_loader: 
        image = image.to(device) 
        image_feat = model.visual_encoder(image)        
        image_embed = model.vision_proj(image_feat[:,0,:])            
        image_embed = F.normalize(image_embed,dim=-1)      
        
        image_feats.append(image_feat)
        image_embeds.append(image_embed)
        torch.cuda.empty_cache()
     
    image_feats = torch.cat(image_feats,dim=0)
    image_embeds = torch.cat(image_embeds,dim=0)
    
    sims_matrix = image_embeds @ text_embeds.t()
    score_matrix_i2t = torch.full((len(data_loader.dataset.image),len(texts)),-100.0).to(device)
    print('Calculating image to text scores')
    for i,sims in enumerate(sims_matrix): 
        topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0) # k_test = 128

        encoder_output = image_feats[i].repeat(config['k_test'],1,1)
        encoder_att = torch.ones(encoder_output.size()[:-1],dtype=torch.long).to(device)
        output = model.text_encoder(encoder_embeds = text_feats[topk_idx], 
                                    attention_mask = text_atts[topk_idx],
                                    encoder_hidden_states = encoder_output,
                                    encoder_attention_mask = encoder_att,                             
                                    return_dict = True,
                                    mode = 'fusion'
                                   )
        score = model.itm_head(output.last_hidden_state[:,0,:])[:,1]
        score_matrix_i2t[i,topk_idx] = score
        
    sims_matrix = sims_matrix.t()
    score_matrix_t2i = torch.full((len(texts),len(data_loader.dataset.image)),-100.0).to(device)
    print('Calculating text to image scores')
    for i,sims in enumerate(sims_matrix): 
        
        topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)
        encoder_output = image_feats[topk_idx]
        encoder_att = torch.ones(encoder_output.size()[:-1],dtype=torch.long).to(device)
        output = model.text_encoder(encoder_embeds = text_feats[i].repeat(config['k_test'],1,1), 
                                    attention_mask = text_atts[i].repeat(config['k_test'],1),
                                    encoder_hidden_states = encoder_output,
                                    encoder_attention_mask = encoder_att,                             
                                    return_dict = True,
                                    mode = 'fusion'
                                   )
        score = model.itm_head(output.last_hidden_state[:,0,:])[:,1]
        score_matrix_t2i[i,topk_idx] = score    
        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluation time {}'.format(total_time_str)) 

    return score_matrix_i2t.cpu(), score_matrix_t2i.cpu(), sims_matrix

@torch.no_grad()
def itm_eval(scores_i2t, scores_t2i, txt2img, img2txt):
    print('scores_i2t:', scores_i2t.shape, 'scores_t2i:', scores_t2i.shape)
    #Images->Text 
    ranks = np.zeros(scores_i2t.shape[0])
    # Calculate the topmost ranks of true texts
    for index,score in enumerate(scores_i2t): 
        inds = np.argsort(score)[::-1] # sorted indices
        # Score
        rank = 1e20
        for i in img2txt[index]:
            tmp = np.where(inds == i)[0][0] # Return the indices of the elements that are non-zero.
            if tmp < rank:
                rank = tmp
        ranks[index] = rank # Store the topmost rank of the i'th text inside img2txt[index]

    # Compute metrics
    tr1 = len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = len(np.where(ranks < 10)[0]) / len(ranks)
  
    #Text->Images 
    ranks = np.zeros(scores_t2i.shape[0])
    
    for index,score in enumerate(scores_t2i):
        inds = np.argsort(score)[::-1]
        ranks[index] = np.where(inds == txt2img[index])[0][0]

    # Compute metrics
    ir1 = len(np.where(ranks < 1)[0]) / len(ranks)
    ir5 = len(np.where(ranks < 5)[0]) / len(ranks)
    ir10 = len(np.where(ranks < 10)[0]) / len(ranks)        

    tr_mean = (tr1 + tr5 + tr10) / 3
    ir_mean = (ir1 + ir5 + ir10) / 3
    r_mean = (tr_mean + ir_mean) / 2

    eval_result =  {'txt_r1': tr1,
                    'txt_r5': tr5,
                    'txt_r10': tr10,
                    'txt_r_mean': tr_mean,
                    'img_r1': ir1,
                    'img_r5': ir5,
                    'img_r10': ir10,
                    'img_r_mean': ir_mean,
                    'r_mean': r_mean}
    return eval_result
