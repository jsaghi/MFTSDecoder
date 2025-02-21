import torch
import torch.nn as nn
import torch.nn.functional as F

# Class DenseFilterExpansion
class DenseFilterExpansion(nn.Module):
  def __init__(self, num_filters, seq_length):
    # Initialize weights and biases
    super().__init__()
    self.seq_length = seq_length
    self.num_filters = num_filters
    self.w = nn.Parameter(torch.randn(self.num_filters, self.seq_length))
    self.b = nn.Parameter(torch.zeros(self.num_filters, self.seq_length))

  # Forward function
  def forward(self, inputs):
    # Slice inputs and weights along the sequence dimension
    in_slice = torch.split(inputs, 1, dim=-1)
    w_slice = torch.split(self.w, 1, dim=-1)
    slices = []
    # Multiply inputs and weights along the sequence dimension
    for i in range(self.seq_length):
      slices.append(torch.matmul(in_slice[i], torch.transpose(w_slice[i], 0, 1)))
    # Stack outputs and remove the dimension added by the stack operation
    outputs = torch.squeeze(torch.stack(slices, axis=-1), dim=1)
    # Add biases and return
    return outputs + self.b
  

# Class FilterAttention
class FilterAttention(nn.Module):
  def __init__(self, num_filters):
    super().__init__()
    # Initialize weights and biases
    self.num_filters = num_filters
    self.query = nn.Parameter(torch.randn(self.num_filters))
    self.key = nn.Parameter(torch.randn(self.num_filters))
    self.value = nn.Parameter(torch.randn(self.num_filters))

  # Forward function
  def forward(self, inputs):
    # Multiply query, key, and value weights by the inputs
    query = torch.mul(inputs, self.query)
    key = torch.mul(inputs, self.key)
    value = torch.mul(inputs, self.value)

    # Calculate attention scores
    attention_scores = torch.multiply(query, key)
    # Calculate attention weights and apply a softmax along the filter dimension
    attention_weights = F.softmax(attention_scores, dim=-2)
    # Multiply values by attention weights
    weighted_value = torch.mul(attention_weights, value)
    # Sum along the filter dimension
    output = torch.sum(weighted_value, dim=-2)
    # Add back the dimension lost by the sum operation and return the output
    output = torch.unsqueeze(output, dim=-2)
    return output

