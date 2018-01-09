import numpy as np
import pickle
import argparse
import ConfigSpace as CS
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.ERROR)
from copy import deepcopy

from smac.facade.smac_facade import SMAC
from smac.scenario.scenario import Scenario

import hpbandster.distributed.utils
from hpbandster.distributed.worker import Worker


def create_config_space():
	cs = CS.ConfigurationSpace()

	adam_final_lr_fraction = CS.UniformFloatHyperparameter("adam_final_lr_fraction",
                                               lower=1 * 10 ** (-4),
                                               default_value=1 * 10 ** (-2),
                                               upper=1.0,
                                               log=True)
	cs.add_hyperparameter(adam_final_lr_fraction)

	adam_initial_lr = CS.UniformFloatHyperparameter("adam_initial_lr",
                                               lower=1 * 10 ** (-4),
                                               upper=1 * 10 ** (-2),
                                               default_value=1 * 10 ** (-3),
                                               log=True)
	cs.add_hyperparameter(adam_initial_lr)

	sgd_final_lr_fraction = CS.UniformFloatHyperparameter("sgd_final_lr_fraction",
                                               lower=1 * 10 ** (-4),
                                               upper=1.0,
                                               default_value=1 * 10 ** (-2),
                                               log=True)
	cs.add_hyperparameter(sgd_final_lr_fraction)

	sgd_initial_lr = CS.UniformFloatHyperparameter("sgd_initial_lr",
                                               lower=1 * 10 ** (-3),
                                               upper=0.5,
                                               default_value=1 * 10 ** (-1),
                                               log=True)
	cs.add_hyperparameter(sgd_initial_lr)

	sgd_momentum = CS.UniformFloatHyperparameter("sgd_momentum",
                                               lower=0.0,
                                               upper=0.99,
                                               default_value=0.9,
                                               log=False)
	cs.add_hyperparameter(sgd_momentum)

	stepDecay_epochs_per_step = CS.UniformIntegerHyperparameter("stepDecay_epochs_per_step",
                                               lower=1,
                                               upper=128,
                                               default_value=16,
                                               log=True)
	cs.add_hyperparameter(stepDecay_epochs_per_step)

	activation = CS.CategoricalHyperparameter("activation", 
  	['relu', 'tanh'],
  	default_value='relu')
	cs.add_hyperparameter(activation)

	batch_size = CS.UniformIntegerHyperparameter("batch_size",
                                               lower=8,
                                               upper=256,
                                               default_value=16,
                                               log=True)
	cs.add_hyperparameter(batch_size)

	dropout_0 = CS.UniformFloatHyperparameter("dropout_0",
                                               lower=0.0,
                                               upper=0.5,
                                               default_value=0.0,
                                               log=False)
	cs.add_hyperparameter(dropout_0)

	dropout_1 = CS.UniformFloatHyperparameter("dropout_1",
                                               lower=0.0,
                                               upper=0.5,
                                               default_value=0.0,
                                               log=False)
	cs.add_hyperparameter(dropout_1)

	dropout_2 = CS.UniformFloatHyperparameter("dropout_2",
                                               lower=0.0,
                                               upper=0.5,
                                               default_value=0.0,
                                               log=False)
	cs.add_hyperparameter(dropout_2)

	dropout_3 = CS.UniformFloatHyperparameter("dropout_3",
                                               lower=0.0,
                                               upper=0.5,
                                               default_value=0.0,
                                               log=False)
	cs.add_hyperparameter(dropout_3)

	l2_reg_0 = CS.UniformFloatHyperparameter("l2_reg_0",
                                               lower=1 * 10 ** (-6),
                                               upper=1 * 10 ** (-2),
                                               default_value=1 * 10 ** (-4),
                                               log=True)
	cs.add_hyperparameter(l2_reg_0)

	l2_reg_1 = CS.UniformFloatHyperparameter("l2_reg_1",
                                               lower=1 * 10 ** (-6),
                                               upper=1 * 10 ** (-2),
                                               default_value=1 * 10 ** (-4),
                                               log=True)
	cs.add_hyperparameter(l2_reg_1)

	l2_reg_2 = CS.UniformFloatHyperparameter("l2_reg_2",
                                               lower=1 * 10 ** (-6),
                                               upper=1 * 10 ** (-2),
                                               default_value=1 * 10 ** (-4),
                                               log=True)
	cs.add_hyperparameter(l2_reg_2)

	l2_reg_3 = CS.UniformFloatHyperparameter("l2_reg_3",
                                               lower=1 * 10 ** (-6),
                                               upper=1 * 10 ** (-2),
                                               default_value=1 * 10 ** (-4),
                                               log=True)
	cs.add_hyperparameter(l2_reg_3)

	learning_rate_schedule = CS.CategoricalHyperparameter("learning_rate_schedule", 
  	['ExponentialDecay', 'StepDecay'],
  	default_value='ExponentialDecay')
	cs.add_hyperparameter(learning_rate_schedule)

	loss_function = CS.CategoricalHyperparameter("loss_function", 
  	['categorical_crossentropy'],
  	default_value='categorical_crossentropy')
	cs.add_hyperparameter(loss_function)

	num_layers = CS.UniformIntegerHyperparameter("num_layers",
                                               lower=1,
                                               upper=4,
                                               default_value=2,
                                               log=False)
	cs.add_hyperparameter(num_layers)

	num_units_0 = CS.UniformIntegerHyperparameter("num_units_0",
                                               lower=16,
                                               upper=256,
                                               default_value=32,
                                               log=True)
	cs.add_hyperparameter(num_units_0)

	num_units_1 = CS.UniformIntegerHyperparameter("num_units_1",
                                               lower=16,
                                               upper=256,
                                               default_value=32,
                                               log=True)
	cs.add_hyperparameter(num_units_1)

	num_units_2 = CS.UniformIntegerHyperparameter("num_units_2",
                                               lower=16,
                                               upper=256,
                                               default_value=32,
                                               log=True)
	cs.add_hyperparameter(num_units_2)

	num_units_3 = CS.UniformIntegerHyperparameter("num_units_3",
                                               lower=16,
                                               upper=256,
                                               default_value=32,
                                               log=True)
	cs.add_hyperparameter(num_units_3)

	optimizer = CS.CategoricalHyperparameter("optimizer",
    ['Adam', 'SGD'],
    default_value='Adam')
	cs.add_hyperparameter(optimizer)

	output_activation = CS.CategoricalHyperparameter("output_activation",
    ['softmax'],
    default_value='softmax')
	cs.add_hyperparameter(output_activation)

	#Conditions.
	cs.add_condition(CS.EqualsCondition(adam_final_lr_fraction, optimizer, 'Adam'))
	cs.add_condition(CS.EqualsCondition(adam_initial_lr, optimizer, 'Adam'))
	cs.add_condition(CS.EqualsCondition(sgd_final_lr_fraction, optimizer, 'SGD'))
	cs.add_condition(CS.EqualsCondition(sgd_initial_lr, optimizer, 'SGD'))
	cs.add_condition(CS.EqualsCondition(sgd_momentum, optimizer, 'SGD'))
	cs.add_condition(CS.EqualsCondition(stepDecay_epochs_per_step, learning_rate_schedule, 'StepDecay'))
	cs.add_condition(CS.GreaterThanCondition(dropout_1, num_layers, 1))
	cs.add_condition(CS.GreaterThanCondition(dropout_2, num_layers, 2))
	cs.add_condition(CS.EqualsCondition(dropout_3, num_layers, 4))
	cs.add_condition(CS.GreaterThanCondition(l2_reg_1, num_layers, 1))
	cs.add_condition(CS.GreaterThanCondition(l2_reg_2, num_layers, 2))
	cs.add_condition(CS.EqualsCondition(l2_reg_3, num_layers, 4))
	cs.add_condition(CS.GreaterThanCondition(num_units_1, num_layers, 1))
	cs.add_condition(CS.GreaterThanCondition(num_units_2, num_layers, 2))
	cs.add_condition(CS.EqualsCondition(num_units_3, num_layers, 4))

	return cs


def objective_function(config, epoch=127, **kwargs):
	# Cast the config to an array such that it can be forwarded to the surrogate
	x = deepcopy(config.get_array())
	x[np.isnan(x)] = -1
	lc = rf.predict(x[None, :])[0]
	c = cost_rf.predict(x[None, :])[0]

	return lc[epoch], {"cost": c, "learning_curve": lc[:epoch].tolist()}


class WorkerWrapper(Worker):
	def compute(self, config, budget, *args, **kwargs):
		cfg = CS.Configuration(cs, values=config)
		loss, info = objective_function(cfg, epoch=int(budget))

		return ({
			'loss': loss,
			'info': {"runtime": info["cost"],
					 "lc": info["learning_curve"]}
		})


if __name__ == '__main__':

	parser = argparse.ArgumentParser()

	parser.add_argument('--run_smac', action='store_true')
	parser.add_argument('--run_hyperband', action='store_true')
	parser.add_argument('--n_iters', default=50, type=int)
	args = vars(parser.parse_args())

	n_iters = args['n_iters']

	cs = create_config_space()
	rf = pickle.load(open("./rf_surrogate_paramnet_mnist.pkl", "rb"))
	cost_rf = pickle.load(open("./rf_cost_surrogate_paramnet_mnist.pkl", "rb"))

	# Begin plotting
	fig_inc = plt.figure()
	ax_inc = fig_inc.add_subplot(111)
	ax_inc.set_xlabel('wall-clock time [s]')
	ax_inc.set_ylabel('incumbent validation error')
	max_inc_perf = 0.0

	if args["run_smac"]:
		scenario = Scenario({"run_obj": "quality",
							 "runcount-limit": n_iters,
							 "cs": cs,
							 "deterministic": "true",
							 "output_dir": ""})

		smac = SMAC(scenario=scenario, tae_runner=objective_function)
		smac.optimize()

		# The following lines extract the incumbent strategy and the estimated wall-clock time of the optimization
		rh = smac.runhistory
		incumbents = []
		incumbent_performance = []
		inc = None
		inc_value = 1
		idx = 1
		t = smac.get_trajectory()

		wall_clock_time = []
		cum_time = 0
		for d in rh.data:
			cum_time += rh.data[d].additional_info["cost"]
			wall_clock_time.append(cum_time)
		for i in range(n_iters):

			if idx < len(t) and i == t[idx].ta_runs - 1:
				inc = t[idx].incumbent
				inc_value = t[idx].train_perf
				idx += 1

			incumbents.append(inc)
			incumbent_performance.append(inc_value)

		# Plot wall clock time and incumbent performance
		max_inc_perf = max(max_inc_perf, max(incumbent_performance))
		ax_inc.plot(wall_clock_time, incumbent_performance, 'ob-', label='SMAC', markersize=2)

		lc_smac = []
		for d in rh.data:
			lc_smac.append(rh.data[d].additional_info["learning_curve"])

		# Plot learning curves
		fig_lc_smac = plt.figure()
		ax_lc_smac = fig_lc_smac.add_subplot(111)
		ax_lc_smac.set_xlabel('Epoch')
		ax_lc_smac.set_ylabel('Validation Error')
		for lc in lc_smac:
			ax_lc_smac.plot(lc, c=np.random.rand(3,))
		fig_lc_smac.savefig('smac_lc.png', dpi=250)

	if args["run_hyperband"]:
		nameserver, ns_port = hpbandster.distributed.utils.start_local_nameserver()

		# starting the worker in a separate thread
		w = WorkerWrapper(nameserver=nameserver, ns_port=ns_port)
		w.run(background=True)

		CG = hpbandster.config_generators.RandomSampling(cs)

		# instantiating Hyperband with some minimal configuration
		HB = hpbandster.HB_master.HpBandSter(
			config_generator=CG,
			run_id='0',
			eta=2,  # defines downsampling rate
			min_budget=1,  # minimum number of epochs / minimum budget
			max_budget=127,  # maximum number of epochs / maximum budget
			nameserver=nameserver,
			ns_port=ns_port,
			job_queue_sizes=(0, 1),
		)
		# runs one iteration if at least one worker is available
		res = HB.run(10, min_n_workers=1)

		# shutdown the worker and the dispatcher
		HB.shutdown(shutdown_workers=True)

		# extract incumbent trajectory and all evaluated learning curves
		traj = res.get_incumbent_trajectory()
		wall_clock_time = []
		cum_time = 0

		for c in traj["config_ids"]:
			cum_time += res.get_runs_by_id(c)[-1]["info"]["runtime"]
			wall_clock_time.append(cum_time)

		lc_hyperband = []
		for r in res.get_all_runs():
			c = r["config_id"]
			lc_hyperband.append(res.get_runs_by_id(c)[-1]["info"]["lc"])

		incumbent_performance = traj["losses"]

		# Plot wall clock time and incumbent performance
		ax_inc.plot(wall_clock_time, incumbent_performance, 'og-', label='Hyperband', markersize=2)
		max_inc_perf = max(max_inc_perf, max(incumbent_performance))

		# Plot learning curves
		fig_lc_hb = plt.figure()
		ax_lc_hb = fig_lc_hb.add_subplot(111)
		ax_lc_hb.set_xlabel('Epoch')
		ax_lc_hb.set_ylabel('Validation Error')
		for lc in lc_hyperband:
			ax_lc_hb.plot(lc, c=np.random.rand(3,))
		fig_lc_hb.savefig('hb_lc.png', dpi=250)

	# Plot incumbent validation erorrs
	ax_inc.set_ylim([0.0, max_inc_perf * 1.1])
	ax_inc.legend(bbox_to_anchor=(1, 1), loc=1, borderaxespad=0.)
	fig_inc.savefig('inc_perf.png', dpi=250)
