# PICME

Scientific initiation (2019 - Ongoing) in Deep Learning at PUC-Rio, supervised by Dr. Matias G delgadino as part of PICME.

The repository includes .ipynb files with implementations in tensorflow and keras created as a study of different types of models.

## Conditional GAN



## Progressive Growing GAN

This was created as a simpler implementation of the network described by (Karras et al., 2017). The idea is to start with small generator and discriminator models which create low resolution images but can be trained faster and later expanded to create higher resolution images. It's expected that the final results should be attained faster and be less unstable than starting with a model that generates images of higher resolutions from scrath.

## Conditional Growing GAN

This was created as combination of the Conditional GAN and Progressive Growing GAN as a proof of concepct of a network that can both generate images of increasing resolution and control the class of the generated images. Both the generator and discriminator were created as to have self-similar architetures irrespective of the size of the model, this was done using the PG-GAN as a basis and including the necessary adaptations to include the label head in the input blocks of both models. This meant a bigger change for the discriminator model as its input dimensions change after every expansion. An alternative would be to grow only the image head of the discriminator, but the resulting architeture would not be self-similar after expanding the model. 

## Reinforcement Learning

This is a simpler implementation of AlphaZero and Monte Carlo Tree Search. This version is trained with the network playing itself and includes episode batching and a combined policy and value model. Originally the model was trained in a Tic-Tac-Toe evironment which was later generalized as a m,n,k-game.

Final results for Tic-Tac-Toe were a X% win rate as player 1 when playing against a random agent and Y% win rate when playing as player 2 againts the same agent using only the network prediction without additional simulations via MCTS. Each episode simulation took 90ms and the entire training was done in about 10 minutes after 10000 episodes. 

Further exploration in more complex environments such as Gomoku freestyle are still under research.
