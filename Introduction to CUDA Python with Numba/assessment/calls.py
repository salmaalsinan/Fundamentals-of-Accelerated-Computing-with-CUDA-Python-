# Use the 'File' menu above to 'Save' after pasting in your 3 function calls.
normalized_gpu=gpu_normalize(greyscales_gpu)
weighted_gpu=gpu_weigh(normalized_gpu, weights_gpu)
SOLUTION=gpu_activate(weighted_gpu)

SOLUTION.copy_to_host()
#SOLUTION
