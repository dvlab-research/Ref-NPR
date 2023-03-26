STYLE=$1 # style_folder
MULTI=${2:-2} # train_length: n*(1+5)
OUTDIR=${3:-./exps/refnpr} # out_dir

export PYTHONPATH=./opt:$PYTHONPATH
export PYTHONPATH=./exps:$PYTHONPATH
# # Step 0: load json, then justify whether to train a base model.
python -W ignore ./exps/ref_pre.py ${STYLE}

# # Step 1: preprocessing step.
python -W ignore ./exps/refnpr/ref_regist.py ${STYLE} --out_dir ${OUTDIR}

# # Step 2: optimization step.
python -W ignore ./exps/refnpr/ref_opt.py ${STYLE} --train_dir ${OUTDIR} \
                --mse_num_epoches `expr $MULTI \* 1` --nnfm_num_epoches `expr $MULTI \* 5` \
                --no_pre_ct --no_post_ct \
                --reset_basis_dim 1

# # Step 3: rendering step.
# python -W ignore ./exps/refnpr/ref_render_circle.py ${OUTDIR} ${STYLE}
python -W ignore ./exps/ref_render.py ${OUTDIR} ${STYLE}\
                 --render_path

# # Step 4: rendering step (train).
python -W ignore ./exps/ref_render.py ${OUTDIR} ${STYLE}\
                --train
