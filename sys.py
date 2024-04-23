# for i in range(time_iterations): 

    #  init pop

"""
    RUN ON DATA:
        for agent in agents:
            for trade in agent.trades: <- do i want to use a binary search here
                if trade.should_close():
                    agent.close_trade(trade) -> updates earnings, num_trades,

            agent.update_inputs() <- update brain with new inputs for this iteration
            action = agent.action_output()
            if agent.action_output() != DO_NOTHING:
                set_target(agent.regression_output())
                do_action(action)

"""