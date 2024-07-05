
import torch
import torch.nn as nn
from transformers.models.longformer.modeling_longformer import (
    LongformerEmbeddings,
    LongformerEncoder,
    LongformerPreTrainedModel
)
from torch.nn import CrossEntropyLoss
from classification_layers import CpaTaskHead


class LongformerMultiPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states, batch_cls_indexes):
        device = hidden_states.device  # Get the device of the input tensor
        pooled_outputs_list = []

        #torch.Size([24, 1283, 768])
        # batch size, seq_len, hidden_state_shape
        #print(hidden_states.shape)

        for sample_idx, sample_cls_indexes in enumerate(batch_cls_indexes):
            indices = sample_cls_indexes.to(device)
            token_tensor = torch.index_select(hidden_states[sample_idx], 0, indices)
            pooled_outputs = self.dense(token_tensor)
            pooled_outputs = self.activation(pooled_outputs)
            pooled_outputs_list.append(pooled_outputs)

        # Convert the list of tensors into a single tensor
        pooled_outputs_tensor = torch.stack(pooled_outputs_list, dim=0)

        return pooled_outputs_tensor
    
class LongformerModelMultiOuput(LongformerPreTrainedModel):
    """
    A Longformer Model with Multi Output.
    """

    def __init__(self, config, add_pooling_layer=True):
        """
        Initializes the LongformerModelMultiOutput.

        Args:
            config (LongformerConfig): Configuration class for Longformer.
            add_pooling_layer (bool, optional): Whether to add a pooling layer. Defaults to True.
        """
        super().__init__(config)
        self.config = config

        # Handling attention window configuration
        if isinstance(config.attention_window, int):
            assert config.attention_window % 2 == 0, "`config.attention_window` has to be an even value"
            assert config.attention_window > 0, "`config.attention_window` has to be positive"
            # Set attention window for each layer if it's a single value
            config.attention_window = [config.attention_window] * config.num_hidden_layers
        else:
            assert len(config.attention_window) == config.num_hidden_layers, (
                "`len(config.attention_window)` should equal `config.num_hidden_layers`. "
                f"Expected {config.num_hidden_layers}, given {len(config.attention_window)}"
            )

        # Initializing components
        self.embeddings = LongformerEmbeddings(config)
        self.encoder = LongformerEncoder(config)
        self.pooler_cta = LongformerMultiPooler(config)

        # Initialize weights
        self.init_weights()

    def _pad_to_window_size(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            token_type_ids: torch.Tensor,
            position_ids: torch.Tensor,
            inputs_embeds: torch.Tensor,
            pad_token_id: int,
    ):
        """
        A helper function to pad tokens and mask to work with implementation of Longformer self-attention.

        Args:
            input_ids (torch.Tensor): Input tensor of token IDs.
            attention_mask (torch.Tensor): Attention mask tensor.
            token_type_ids (torch.Tensor): Token type IDs tensor.
            position_ids (torch.Tensor): Position IDs tensor.
            inputs_embeds (torch.Tensor): Input embeddings tensor.
            pad_token_id (int): Padding token ID.

        Returns:
            tuple: Padding length, padded tensors.
        """

        # Determine attention window size
        attention_window = (
            self.config.attention_window
            if isinstance(self.config.attention_window, int)
            else max(self.config.attention_window)
        )

        # Ensure attention window is even
        assert attention_window % 2 == 0, f"`attention_window` should be an even value. Given {attention_window}"
        input_shape = input_ids.shape if input_ids is not None else inputs_embeds.shape
        batch_size, seq_len = input_shape[:2]

        # Calculate padding length to ensure window alignment
        padding_len = (attention_window - seq_len % attention_window) % attention_window

        if padding_len > 0:
            if input_ids is not None:
                # Pad input IDs with pad_token_id
                input_ids = nn.functional.pad(input_ids, (0, padding_len), value=pad_token_id)
            if position_ids is not None:
                # Pad position IDs with pad_token_id
                position_ids = nn.functional.pad(position_ids, (0, padding_len), value=pad_token_id)
            if inputs_embeds is not None:
                # Generate padding embeddings
                input_ids_padding = inputs_embeds.new_full(
                    (batch_size, padding_len),
                    self.config.pad_token_id,
                    dtype=torch.long,
                )
                # Get padding embeddings
                inputs_embeds_padding = self.embeddings(input_ids_padding)
                # Concatenate original embeddings with padding embeddings
                inputs_embeds = torch.cat([inputs_embeds, inputs_embeds_padding], dim=-2)

            # Pad attention_mask to align with new sequence length
            attention_mask = nn.functional.pad(
                attention_mask, (0, padding_len), value=0
            )  # no attention on the padding tokens
            # Pad token_type_ids with 0
            token_type_ids = nn.functional.pad(token_type_ids, (0, padding_len), value=0)

        return padding_len, input_ids, attention_mask, token_type_ids, position_ids, inputs_embeds

    def _merge_to_attention_mask(self, attention_mask: torch.Tensor, global_attention_mask: torch.Tensor):
        """
        Merges local and global attention masks.

        Args:
            attention_mask (torch.Tensor): Local attention mask.
            global_attention_mask (torch.Tensor): Global attention mask.

        Returns:
            torch.Tensor: Merged attention mask.
        """
        # longformer self attention expects attention mask to have 0 (no attn), 1 (local attn), 2 (global attn)
        # (global_attention_mask + 1) => 1 for local attention, 2 for global attention
        # => final attention_mask => 0 for no attention, 1 for local attention 2 for global attention
        if attention_mask is not None:
            attention_mask = attention_mask * (global_attention_mask + 1)
        else:
            # simply use `global_attention_mask` as `attention_mask`
            # if no `attention_mask` is given
            attention_mask = global_attention_mask + 1
        return attention_mask

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            global_attention_mask=None,
            head_mask=None,
            token_type_ids=None,
            position_ids=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            cls_indexes=None
    ):
        """
        Forward pass of the Longformer Model.

        Args:
            input_ids (torch.Tensor, optional): Input token IDs. Defaults to None.
            attention_mask (torch.Tensor, optional): Attention mask. Defaults to None.
            global_attention_mask (torch.Tensor, optional): Global attention mask. Defaults to None.
            head_mask (torch.Tensor, optional): Head mask. Defaults to None.
            token_type_ids (torch.Tensor, optional): Token type IDs. Defaults to None.
            position_ids (torch.Tensor, optional): Position IDs. Defaults to None.
            inputs_embeds (torch.Tensor, optional): Input embeddings. Defaults to None.
            output_attentions (bool, optional): Whether to output attentions. Defaults to None.
            output_hidden_states (bool, optional): Whether to output hidden states. Defaults to None.
            return_dict (bool, optional): Whether to return a dictionary. Defaults to None.

        Raises:
            ValueError: If both input_ids and inputs_embeds are provided.

        Returns:
            tuple: Model outputs.
        """

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            # Issue warning if no attention mask provided
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # merge `global_attention_mask` and `attention_mask`
        if global_attention_mask is not None:
            attention_mask = self._merge_to_attention_mask(attention_mask, global_attention_mask)

        padding_len, input_ids, attention_mask, token_type_ids, position_ids, inputs_embeds = self._pad_to_window_size(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            pad_token_id=self.config.pad_token_id,
        )

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)[:, 0, 0,
                                                :]

        # Get embeddings
        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )

        # Pass through encoder
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            padding_len=padding_len,
            output_hidden_states=True
        )
        # Pool the outputs
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler_cta(sequence_output, cls_indexes)  # (N_CLS, 768): N_CLS number of cls tokens


        #last_four_hidden_states = encoder_outputs[1][-4:]
        #last_four_hidden_states_tensor = torch.stack(last_four_hidden_states, dim=0)
        #pooled_output = self.pooler_cta(last_four_hidden_states_tensor, cls_indexes)  # (N_CLS, 768): N_CLS number of cls tokens

        return pooled_output  # spooled_output

class LongformerForMultiOutputClassification(LongformerPreTrainedModel):
    """
    Longformer Model for Multi-Output Classification.
    """

    def __init__(self, config):
        """
        Initializes the LongformerForMultiOutputClassification.

        Args:
            config (LongformerConfig): Configuration class for Longformer.
        """
        super().__init__(config)

        # Initialize Longformer model
        self.longformer = LongformerModelMultiOuput(config)
        self.dropout = nn.Dropout(0.1)
        self.init_weights()
        self.classifier = CpaTaskHead(config.hidden_size, config.num_labels, config.task_specific_params["num_cls"])
        self.loss_fn = CrossEntropyLoss() 
        self.cls_positions = torch.tensor(config.task_specific_params["cls_positions"])

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            global_attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            cls_indexes=None
    ):
        """
        Forward pass of the Longformer Model for Multi-Output Classification.

        Args:
            input_ids (torch.Tensor, optional): Input token IDs. Defaults to None.
            attention_mask (torch.Tensor, optional): Attention mask. Defaults to None.
            global_attention_mask (torch.Tensor, optional): Global attention mask. Defaults to None.
            token_type_ids (torch.Tensor, optional): Token type IDs. Defaults to None.
            position_ids (torch.Tensor, optional): Position IDs. Defaults to None.
            head_mask (torch.Tensor, optional): Head mask. Defaults to None.
            inputs_embeds (torch.Tensor, optional): Input embeddings. Defaults to None.
            labels (torch.Tensor, optional): Labels for classification. Defaults to None.

        Returns:
            tuple: Model outputs.
        """
        batch_size = input_ids.shape[0]
        cls_indexes = torch.stack([self.cls_positions for _ in range(batch_size)])

        pooled_output = self.longformer(
            input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            cls_indexes=cls_indexes,
        )

        pooled_output = self.dropout(pooled_output)

        logits = self.classifier(pooled_output)

        loss = self.loss_fn(logits, labels.float())

        return {"loss": loss, "logits": logits}


def prepare_model(num_cta_classes, num_cls):

    cls_indexes_cta = [i * 513 for i in range(num_cls)]    
    model = LongformerForMultiOutputClassification.from_pretrained('allenai/longformer-base-4096',\
                                             num_labels=num_cta_classes,\
                                             task_specific_params={"num_cls":num_cls, "cls_positions":cls_indexes_cta})

    return model
