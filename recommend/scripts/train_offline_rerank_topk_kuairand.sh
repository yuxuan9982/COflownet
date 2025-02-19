mkdir -p output

mkdir -p output/kuairand_1k/
mkdir -p output/kuairand_1k/agent

env_path="output/kuairand_1k/env/"
env_model_path=${env_path}log/user_KRMBUserResponse_MaxOut_lr0.0001_reg0.1.model.log
output_path="output/kuairand_1k/agent/"

# environment arguments
ENV_CLASS='KRUserEnvironment_ListRec'
SLATE_SIZE=6
MAX_STEP=5
TEMPER=5
EP_BS=128
RHO=0.1

READER_CLASS='KRMBSlateReader'
N_TEST=4

# policy arguments
POLICY_CLASS='TwoStageOnlinePolicy'
INIT_LIST_SIZE=50
INIT_POS_OFFSET=0.1
INIT_NEG_OFFSET=0.1
INIT_LOSS_COEF=0.1
INIT_N_NEG=100
LOSS='Reward BCE'

# agent arguments
AGENT_CLASS='OfflineRerankAgentWithOnlineTest'
N_ITER=5000
INITEP=0.1
ELBOW=0.1
BS=128
REWARD_FUNC='get_immediate_reward'
EXPLORE=1.0

# buffer arguments
BUFFER_CLASS='SequentialBuffer'
RAND_RATE=0.5


for REG in 0 # 0.00001 # 0.001 0.0001
do
    for ACTOR_LR in 0.0001 # 0.001
    do
        for SEED in 13 17 19 23 29
        do
            file_key=${AGENT_CLASS}_${POLICY_CLASS}_actor${ACTOR_LR}_niter${N_ITER}_reg${REG}_ep${INITEP}_bs${BS}_epbs${EP_BS}_seed${SEED}

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
                --initial_list_size ${INIT_LIST_SIZE}\
                --stage1_n_neg ${INIT_N_NEG}\
                --stage1_state2z_hidden_dims 128\
                --stage1_pos_offset ${INIT_POS_OFFSET}\
                --stage1_neg_offset ${INIT_NEG_OFFSET}\
                --initial_loss_coef ${INIT_LOSS_COEF}\
                --loss ${LOSS}\
                --user_latent_dim 16\
                --item_latent_dim 16\
                --transformer_enc_dim 32\
                --transformer_n_head 4\
                --transformer_d_forward 64\
                --transformer_n_layer 2\
                --state_hidden_dims 128\
                --dropout_rate 0.1\
                --agent_class ${AGENT_CLASS}\
                --learn_initial_during_rerank\
                --reward_func ${REWARD_FUNC}\
                --n_iter ${N_ITER}\
                --train_every_n_step 1\
                --start_train_at_step 100\
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