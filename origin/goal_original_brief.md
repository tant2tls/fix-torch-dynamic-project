1.  in origin torch 2.3.1 code, if we compile some weird operations -> it will raise the error File "/prj/corp/airesearch/lasvegas/vol22-scratch/users/tanngo/PhD/fix_torch_compile/optimizer-image/genai-sd-optimization/venv_image_server_onediff/lib/python3.11/site-packages/torch/_inductor/index_propagation.py", line 62, in constant
    expr = sympy.Float(float(value))
                       ^^^^^^^^^^^^
  File "/prj/corp/airesearch/lasvegas/vol22-scratch/users/tanngo/PhD/fix_torch_compile/optimizer-image/genai-sd-optimization/venv_image_server_onediff/lib/python3.11/site-packages/sympy/core/expr.py", line 375, in __float__
    raise TypeError("Cannot convert expression to float")"
2. help me to create code to install torch 2.3.1 then compile real bug operation -> show explanation -> then show guides how to fix and run compile weired operations again to see successlly
3. This project aim to show my debug skills which will be used for PhD application CS AI in top ML system lab in US -> You must try to replicate the real bug and solution -> write readme.md clearly how to install torch 2.3.1 and neccessary packages (no need to onediff or some packaages using for inferences, we only focus on real bug operations that torch 2.3.1 misunderstand)
4. you can see ./small_logs.txt for a part of log -> you should run again with( then run bash ./warmup_simple.sh to see the errors): 
source venv_image_server_onediff/bin/activate && \
CUDA_VISIBLE_DEVICES=0,1 WORKER_NETWORK_PORT=20021 \
ACTIVATED_TASKS='["dreamshaper-text-to-image", "dreamshaper-image-to-image"]' \
torchrun --master_port=20021 --rdzv_id=1 --rdzv_backend=c10d --nproc_per_node=2 \
         --rdzv_endpoint=localhost:29503 worker_server.py --gpu-only

- we only create our new repo because origin genai-sd-optimzation is internall project, we will focus on fix_torch_dynamic_project (don't relevant with sd,flux comfyui inference logic in this repo)