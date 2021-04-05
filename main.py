
import logging
import tensorflow as tf
from tqdm import tqdm
from DRLCM.agent import Agent
from DRLCM.DRLCMconfig import get_config
from DRLCM.EnvironmentChanger import *
from DRLCM.environment import *




if __name__ == "__main__":

    print("Warning ! If you have changed the inputs you need to specify new initial Placement in config file!!! ")

    config, _ =  get_config()
    env_input = get_Input_parameters()
    env = Environment( env_input)
    Envchange = EnvironmentChanger()

    """ Agent """
    state_size_sequence = config.Length_plc_chain
    state_size_embeddings = config.Length_plc_chain
    action_size = config.Length_plc_chain
    agent = Agent(state_size_embeddings, state_size_sequence, action_size, config.batch_size, config.learning_rate, config.hidden_dim, config.num_stacks, num_Total_cache_nodes,num_non_safety_contents)


    model_snapshot = [var for var in tf.global_variables()]
    saver = tf.train.Saver(var_list= model_snapshot, keep_checkpoint_every_n_hours=1.0)

    print("restoring already built DRLCM model")
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        if config.load_model:
            saver.restore(sess, "SavedModel/tf_DRLCM.ckpt")
            print("the saved DRLCM Model is restored.")

        
        if config.train_mode:

            print("\n training is started :")
            for e in tqdm(range(config.num_epoch)):


                state = Envchange.getNewState()
                input_state = vector_embedding(services)

                # Compute placement
                feed = {agent.input_: input_state, agent.input_len_: [item for item in services.serviceLength]}
                a = agent.ptr.positions
                positions = sess.run(agent.ptr.positions, feed_dict=feed)

                reward = np.zeros(config.batch_size)

                # Compute environment
                for batch in range(config.batch_size):
                    env.clear()
                    env.step(positions[batch], services.state[batch], services.serviceLength[batch])
                    reward[batch] = env.reward

                # RL Learning
                feed = {agent.reward_holder: [item for item in reward], agent.positions_holder: positions,
                        agent.input_: input_state, agent.input_len_: [item for item in services.serviceLength]}

                summary, _ = sess.run([agent.merged, agent.train_step], feed_dict=feed)

                if e % 100 == 0:
                    print("\n Mean batch ", e, "reward:", np.mean(reward))
                    writer.add_summary(summary, e)

                # Save intermediary model variables
                if config.save_model and e % max(1, int(config.num_epoch / 5)) == 0 and e != 0:
                    save_path = saver.save(sess, "save/tmp.ckpt", global_step=e)
                    print("\n Model saved in file: %s" % save_path)

                e += 1

            print("\n Training COMPLETED!")

            if config.save_model:
                save_path = saver.save(sess, "save/tf_binpacking.ckpt")
                print("\n Model saved in file: %s" % save_path)

        # Testing
        else:
            state = Envchange.getNewState()
            services.getNewState()

            # Vector embedding
            input_state = vector_embedding(services)

            # Compute placement
            feed = {agent.input_: input_state, agent.input_len_: [item for item in services.serviceLength]}
            a = agent.ptr.positions
            positions = sess.run(agent.ptr.positions, feed_dict=feed)

            reward = np.zeros(config.batch_size)

            # Compute environment
            for batch in range(config.batch_size):
                env.clear()
                env.step(positions[batch], services.state[batch], services.serviceLength[batch])
                reward[batch] = env.reward

                # Render some batch services
                if batch % max(1, int(config.batch_size / 5)) == 0:
                    print("\n Rendering batch ", batch, "...")
                    env.render(batch)

            # Calculate performance
            if config.enable_performance:

                print("\n Calculating optimal solutions... ")
                optReward = np.zeros(config.batch_size)

                for batch in tqdm(range(config.batch_size)):
                    optPlacement = solver(services.state[batch], services.serviceLength[batch], env)
                    env.clear()
                    env.step(optPlacement, services.state[batch], services.serviceLength[batch])
                    optReward[batch] = env.reward
                    assert optReward[batch] + 0.1 > reward[batch]  # Avoid inequalities in the last decimal...

                performance = np.sum(reward) / np.sum(optReward)
                print("\n Performance: ", performance)




def get_Input_parameters():
    # costomize this function so that your selected input features get to an instance of the environment
    pass
