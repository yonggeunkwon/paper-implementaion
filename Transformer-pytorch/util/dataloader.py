import torch
from torch.utils.data import Dataset
from typing import Any

class BilingualDataset(Dataset):

    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len) -> None:
        super().__init__()

        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len

        # special token 정의하기
        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)

    
    def __len__(self):
        return len(self.ds)
    
    def casual_mask(self, size):
        # Creating a square matrix of dimensions 'size x size' filled with ones
        mask = torch.triu(torch.ones(1, size, size), diagonal = 1).type(torch.int)  # torch.Size([1, 350, 350])
        return mask == 0
    
    def __getitem__(self, index: Any) -> Any:
        print("-------start---------")
        src_target_pair = self.ds[index]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        # Tokenizing source and target texts
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids
        # print("src_text : ", src_text)
        # print("self.seq_len" ,self.seq_len)
        # print("enc_input_tokens : ", enc_input_tokens)

        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError('Sentence is too long')
        
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype = torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype = torch.int64)
            ]
        )
        # print("self.sos_token :", self.sos_token)
        # print("2 :", torch.tensor(enc_input_tokens, dtype = torch.int64))
        # print("2.shape :", torch.tensor(enc_input_tokens, dtype = torch.int64).shape)
        # print("self.eos_token :", self.eos_token)
        # print("4 :", torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype = torch.int64))
        # print("4.shape :", torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype = torch.int64).shape)

        # print("encoder_input : ", encoder_input)
        # print("encoder_input.shape : " ,encoder_input.shape)

        decoder_input = torch.cat(
            [
                self.sos_token,  
                torch.tensor(dec_input_tokens, dtype = torch.int64), 
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype = torch.int64) 
            ]
        
        )

        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype = torch.int64), # Inserting the tokenized target text
                self.eos_token, # Inserting the '[EOS]' token 
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype = torch.int64) # Adding padding tokens
                
            ]
        )
        
        # Ensuring that the length of each tensor above is equal to the defined 'seq_len'
        # 거짓이라면 assertion error 발생
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len
        
        return {
            'encoder_input': encoder_input,
            'decoder_input': decoder_input, 
            'encoder_mask': (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),
            'decoder_mask': (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & self.casual_mask(decoder_input.size(0)), 
            'label': label,
            'src_text': src_text,
            'tgt_text': tgt_text
        }    
