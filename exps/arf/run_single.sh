STYLE=$1 # style_name
MULTI=${2:-2} # train_length: n*(1+5)
OUTDIR=${3:-./exps/arf} # out_dir

export PYTHONPATH=./opt:$PYTHONPATH
export PYTHONPATH=./exps:$PYTHONPATH
# Step 0: load json, then justify whether to train a base model.
python -W ignore ./exps/ref_pre.py ${STYLE}

# Step 1: optimization step.
python -W ignore ./exps/arf/arf_opt.py ${STYLE} --train_dir ${OUTDIR} \
                --mse_num_epoches `expr $MULTI \* 1` --nnfm_num_epoches `expr $MULTI \* 5` \
                --reset_basis_dim 1

# Step 2: rendering step.
# python -W ignore ./exps/arf/ref_render_circle.py ${OUTDIR} ${STYLE}
python -W ignore ./exps/ref_render.py ${OUTDIR} ${STYLE}\
                --render_path

# Step 3: rendering step (train).
python -W ignore ./exps/ref_render.py ${OUTDIR} ${STYLE}\
                --train
