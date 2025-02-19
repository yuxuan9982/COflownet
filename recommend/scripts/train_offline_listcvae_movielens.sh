mkdir -p output

mkdir -p output/ml1m/
mkdir -p output/ml1m/agent

env_path="output/ml1m/env/"
env_model_path=${env_path}log/user_KRMBUserResponse_MaxOut_lr0.0001_reg0.01.model.log
output_path="output/ml1m/agent/"

# environment arguments
ENV_CLASS='MLUserEnvironment_ListRec'
SLATE_SIZE=6
MAX_STEP=5
TEMPER=5
EP_BS=128
RHO=0.2

READER_CLASS='MLSlateReader'
N_TEST=1

# policy arguments
POLICY_CLASS='ListCVAE'
BETA=1.0
R_TARGET=2.5
LOSS='ELBO'

# agent arguments
AGENT_CLASS='OfflineRerankAgentWithOnlineTest'
N_ITER=10000
INITEP=0.05
ELBOW=0.1
BS=128
REWARD_FUNC='get_immediate_reward'
EXPLORE=1.0

# buffer arguments
# BUFFER_CLASS='BaseBuffer'
BUFFER_CLASS='SequentialBuffer'
RAND_RATE=0.5

for REG in 0.0001 # 0.00001 0.0001
do
    for R_TARGET in 2.0
    do
        for ACTOR_LR in 0.00001
        do
            for BETA in 1.0
            do
                for SEED in 11 13 17 19 23
                do
                    file_key=${AGENT_CLASS}_${POLICY_CLASS}_beta${BETA}_rt${R_TARGET}_actor${ACTOR_LR}_niter${N_ITER}_reg${REG}_ep${INITEP}_bs${BS}_epbs${EP_BS}_seed${SEED}

                    mkdir -p ${output_path}/${file_key}/

                    python train_online_policy.py\
                        --seed ${SEED}\
                        --cuda 0\
                        --env_class ${ENV_CLASS}\
                        --uirm_log_path ${env_model_path}\
                        --initial_temper ${TEMPER}\
                        --slate_size ${SLATE_SIZE}\
                        --max_step_per_episode ${MAX_STEP}\
                        --episode_batch_size ${EP_BS}\
                        --item_correlation ${RHO}\
                        --new_reader_class ${READER_CLASS}\
                        --env_test_holdout ${N_TEST}\
                        --policy_class ${POLICY_CLASS}\
                        --loss ${LOSS}\
                        --cvae_beta ${BETA}\
                        --target_reward ${R_TARGET}\
                        --cvae_c_dim 16\
                        --cvae_z_dim 32\
                        --cvae_prior_hidden_dims 128\
                        --cvae_encoder_hidden_dims 256\
                        --cvae_decoder_hidden_dims 256\
                        --gfn_state_hidden_dims 128\
                        --gfn_flowzero_hidden_dims 128\
                        --user_latent_dim 16\
                        --item_latent_dim 16\
                        --transformer_enc_dim 32\
                        --transformer_n_head 4\
                        --transformer_d_forward 64\
                        --transformer_n_layer 2\
                        --state_hidden_dims 128\
                        --dropout_rate 0.1\
                        --agent_class ${AGENT_CLASS}\
                        --reward_func ${REWARD_FUNC}\
                        --n_iter ${N_ITER}\
                        --train_every_n_step 1\
                        --start_train_at_step 10\
                        --initial_greedy_epsilon ${INITEP}\
                        --final_greedy_epsilon 0.01\
                        --elbow_greedy ${ELBOW}\
                        --check_episode 20\
                        --test_episode 20\
                        --save_episode 200\
                        --save_path ${output_path}/${file_key}/model\
                        --batch_size ${BS}\
                        --actor_lr ${ACTOR_LR}\
                        --actor_decay ${REG}\
                        --explore_rate ${EXPLORE}\
                        --buffer_class ${BUFFER_CLASS}\
                        --buffer_size 50000\
                        --random_rate ${RAND_RATE}\
                        > ${output_path}/${file_key}/log
                done
            done
        done
    done
done



# for REG in 0 # 0.00001 0.0001
# do
#     for BETA in 0.03 0.1 0.3 0.003
#     do
#         for SEED in 13 # 13 17 19 23
#         do
#             for ACTOR_LR in 0.0001 # 0.0003 0.001
#             do
#                 file_key=${AGENT_CLASS}_${POLICY_CLASS}_beta${BETA}_rt${R_TARGET}_actor${ACTOR_LR}_niter${N_ITER}_reg${REG}_ep${INITEP}_bs${BS}_epbs${EP_BS}_seed${SEED}

#                 mkdir -p ${output_path}/${file_key}/

#                 python train_online_policy.py\
#                     --seed ${SEED}\
#                     --cuda 0\
#                     --env_class ${ENV_CLASS}\
#                     --uirm_log_path ${env_model_path}\
#                     --initial_temper ${TEMPER}\
#                     --slate_size ${SLATE_SIZE}\
#                     --max_step_per_episode ${MAX_STEP}\
#                     --episode_batch_size ${EP_BS}\
#                     --item_correlation ${RHO}\
#                     --policy_class ${POLICY_CLASS}\
#                     --loss ${LOSS}\
#                     --cvae_beta ${BETA}\
#                     --target_reward ${R_TARGET}\
#                     --cvae_c_dim 16\
#                     --cvae_z_dim 32\
#                     --cvae_prior_hidden_dims 128\
#                     --cvae_encoder_hidden_dims 256\
#                     --cvae_decoder_hidden_dims 256\
#                     --gfn_state_hidden_dims 128\
#                     --gfn_flowzero_hidden_dims 128\
#                     --user_latent_dim 16\
#                     --item_latent_dim 16\
#                     --transformer_enc_dim 32\
#                     --transformer_n_head 4\
#                     --transformer_d_forward 64\
#                     --transformer_n_layer 2\
#                     --state_hidden_dims 128\
#                     --dropout_rate 0.1\
#                     --agent_class ${AGENT_CLASS}\
#                     --reward_func ${REWARD_FUNC}\
#                     --n_iter ${N_ITER}\
#                     --train_every_n_step 1\
#                     --start_train_at_step 10\
#                     --initial_greedy_epsilon ${INITEP}\
#                     --final_greedy_epsilon 0.01\
#                     --elbow_greedy ${ELBOW}\
#                     --check_episode 20\
#                     --test_episode 20\
#                     --save_episode 200\
#                     --save_path ${output_path}/${file_key}/model\
#                     --batch_size ${BS}\
#                     --actor_lr ${ACTOR_LR}\
#                     --actor_decay ${REG}\
#                     --explore_rate ${EXPLORE}\
#                     --buffer_class ${BUFFER_CLASS}\
#                     --buffer_size 50000\
#                     --random_rate ${RAND_RATE}\
#                     > ${output_path}/${file_key}/log
#             done
#         done
#     done
# done



# for R_TARGET in 1.5 2.5 3.0 # 0.00001 0.0001
# do
#     for BETA in 0.01 # 0.03 0.1 
#     do
#         for SEED in 13 # 13 17 19 23
#         do
#             for ACTOR_LR in 0.0001 # 0.0003 0.001
#             do
#                 file_key=${AGENT_CLASS}_${POLICY_CLASS}_beta${BETA}_rt${R_TARGET}_actor${ACTOR_LR}_niter${N_ITER}_reg${REG}_ep${INITEP}_bs${BS}_epbs${EP_BS}_seed${SEED}

#                 mkdir -p ${output_path}/${file_key}/

#                 python train_online_policy.py\
#                     --seed ${SEED}\
#                     --cuda 0\
#                     --env_class ${ENV_CLASS}\
#                     --uirm_log_path ${env_model_path}\
#                     --initial_temper ${TEMPER}\
#                     --slate_size ${SLATE_SIZE}\
#                     --max_step_per_episode ${MAX_STEP}\
#                     --episode_batch_size ${EP_BS}\
#                     --item_correlation ${RHO}\
#                     --policy_class ${POLICY_CLASS}\
#                     --loss ${LOSS}\
#                     --cvae_beta ${BETA}\
#                     --target_reward ${R_TARGET}\
#                     --cvae_c_dim 16\
#                     --cvae_z_dim 32\
#                     --cvae_prior_hidden_dims 128\
#                     --cvae_encoder_hidden_dims 256\
#                     --cvae_decoder_hidden_dims 256\
#                     --gfn_state_hidden_dims 128\
#                     --gfn_flowzero_hidden_dims 128\
#                     --user_latent_dim 16\
#                     --item_latent_dim 16\
#                     --transformer_enc_dim 32\
#                     --transformer_n_head 4\
#                     --transformer_d_forward 64\
#                     --transformer_n_layer 2\
#                     --state_hidden_dims 128\
#                     --dropout_rate 0.1\
#                     --agent_class ${AGENT_CLASS}\
#                     --reward_func ${REWARD_FUNC}\
#                     --n_iter ${N_ITER}\
#                     --train_every_n_step 1\
#                     --start_train_at_step 10\
#                     --initial_greedy_epsilon ${INITEP}\
#                     --final_greedy_epsilon 0.01\
#                     --elbow_greedy ${ELBOW}\
#                     --check_episode 20\
#                     --test_episode 20\
#                     --save_episode 200\
#                     --save_path ${output_path}/${file_key}/model\
#                     --batch_size ${BS}\
#                     --actor_lr ${ACTOR_LR}\
#                     --actor_decay ${REG}\
#                     --explore_rate ${EXPLORE}\
#                     --buffer_class ${BUFFER_CLASS}\
#                     --buffer_size 50000\
#                     --random_rate ${RAND_RATE}\
#                     > ${output_path}/${file_key}/log
#             done
#         done
#     done
# done



