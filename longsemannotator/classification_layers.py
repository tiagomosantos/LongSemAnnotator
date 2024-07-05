import torch
import torch.nn as nn

class CpaTaskHead(nn.Module):
    def __init__(self, input_size, output_size, num_chunks):
        super(CpaTaskHead, self).__init__()
        self.num_chunks = num_chunks
        self.A = nn.Linear(input_size, output_size, bias=False)
        self.B = nn.Linear(input_size, output_size, bias=False)

    def forward(self, x):
        # Split the input tensor into num_chunks along the second dimension
        chunks = torch.chunk(x, chunks=self.num_chunks, dim=1)
        
        # Process the first chunk with linear layer A
        y1 = self.A(chunks[0].squeeze(1))
        
        # Initialize the output by adding the result of the first chunk
        y_combined = y1
        
        # Process the remaining chunks with linear layer B and add to the output
        for chunk in chunks[1:]:
            y_combined += self.B(chunk.squeeze(1))
        
        return y_combined

class CpaTaskHead2(nn.Module):
    def __init__(self, input_size, output_size):
        super(CpaTaskHead, self).__init__()
        self.A = nn.Linear(input_size, output_size, bias=False)
        self.B = nn.Linear(input_size, output_size, bias=False)
        self.C = nn.Linear(input_size, output_size, bias=False)
    
    def forward(self, x):
        # Split the input tensor along the second dimension
        x1, x2, x3 = torch.chunk(x, chunks=3, dim=1)
        # Pass each input tensor through its corresponding linear transformation
        y1 = self.A(x1.squeeze(1))
        y2 = self.B(x2.squeeze(1))
        y3 = self.C(x3.squeeze(1))

        # Combine the results with trainable weights
        y = y1 +  y2 +  y3

        return y