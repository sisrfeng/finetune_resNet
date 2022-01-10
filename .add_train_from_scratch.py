
# from scratch  vs  finetune
# Initialize the non-pretrained version of the model used for this run
model_scratch, _ = initialize_model(model_name,
                                    num_classes,
                                    freeze01=False,
                                    use_pretrained=False
                                    )

model_scratch = model_scratch.to(myXPU)
_,scratch_hist = train_model(model_scratch,
                            dataloaders_dict,
                            loss__,
                            optim.SGD(model_scratch.parameters(), lr=0.001, momentum=0.9),
                            # num_epochs=num_epochs,
                            num_epochs=num_epochs,
                            is_inception=(model_name=="inception"))