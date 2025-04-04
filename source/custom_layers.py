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
    outputs = torch.squeeze(torch.stack(slices, dim=-1), dim=1)
    # Add biases and return
    return outputs + self.b
  

# Class FilterAttention
class FilterAttention(nn.Module):
  def __init__(self, num_filters, seq_length):
    super().__init__()
    # Initialize weights and biases
    self.num_filters = num_filters
    self.seq_length = seq_length
    self.query = nn.Parameter(torch.randn(self.num_filters, self.seq_length))
    self.key = nn.Parameter(torch.randn(self.num_filters, self.seq_length))
    self.value = nn.Parameter(torch.randn(self.num_filters, self.seq_length))

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


# Class for convolutional filter with insertion. Filter iterates over input,
# then the output of the filter is inserted into the input at intervals of
# kernel_size + 1
class ConvInsert(nn.Module):
  def __init__(self, input_size, kernel_size):
    super().__init__()
    # Save input kernel size and use it to calculate the output size
    self.kernel_size = kernel_size
    self.output_length = (4 * input_size) // kernel_size - 2

    # Initialize weights and biases for the two kernels
    self.w1 = nn.Parameter(torch.randn(self.kernel_size, 1))
    self.w2 = nn.Parameter(torch.randn(self.kernel_size, 1))
    self.b = nn.Parameter(torch.zeros(self.output_length, 1))

  # Forward method of the custom module
  def forward(self, inputs):
    # Split the input into equal chunks of half the length of the kernel size
    in_slices = torch.split(inputs, self.kernel_size // 2, -1)
    # Create a list to store new output slices
    out_slices = []

    # Iterate through each slice of the input sequence
    for i in range(len(in_slices) - 1):
      # Combine adjacent input slices so they can be operated on by the kernels
      super_slice = torch.cat((in_slices[i], in_slices[i + 1]), dim=-1)

      # Calculate outputs whcih while spliced into the original sequence
      out1 = torch.matmul(super_slice, self.w1)
      out2 = torch.matmul(super_slice, self.w2)
      out1 += self.b[2 * i]
      out2 += self.b[2 * i + 1]

      # Concatenate the outputs
      out = torch.cat((out1, out2), dim=-1)

      # Concatenate the one of the input slices with the outputs
      out_slices.append(torch.cat((in_slices[i], out), dim=-1))
      # When the end of the input sequence is reached, concatenate the last input
      # slice without adding an output
      if i == len(in_slices) - 2:
        out_slices.append(in_slices[i + 1])
    
    # Concatenate and return the output slices in a single tensor
    outputs = torch.cat(out_slices, dim=-1)
    return outputs
  