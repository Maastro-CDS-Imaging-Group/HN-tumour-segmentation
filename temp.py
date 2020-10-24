from trainutils.trainer import Trainer


training_config, validation_config, logging_config = None, None, None

trainer = Trainer(training_config, validation_config, logging_config)

model = None

trainer.train(model)
