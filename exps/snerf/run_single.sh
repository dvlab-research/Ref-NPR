STYLE=$1 # style_name
MULTI=${2:-1} # train_length: n*(2+5)
OUTDIR=${3:-./exps/snerf} # out_dir

export PYTHONPATH=./opt:$PYTHONPATH

# Step 0: load json, then justify whether to train a base model.
python -W ignore ./exps/ref_pre.py ${STYLE}

# Step 1: optimization step.
python -W ignore ./exps/snerf/snerf_opt.py ${STYLE} --train_dir ${OUTDIR} \
                --mse_num_epoches `expr $MULTI \* 2` --nnfm_num_epoches `expr $MULTI \* 5` \
                --no_pre_ct --no_post_ct \
                --reset_basis_dim 1

# Step 2: rendering step.
# python -W ignore ./exps/ref_render_circle.py ${OUTDIR} ${STYLE}
python -W ignore ./exps/ref_render.py ${OUTDIR} ${STYLE}\
                --render_path

# Step 3: rendering step (train).
python -W ignore ./exps/ref_render.py ${OUTDIR} ${STYLE}\
                --train
