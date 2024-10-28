import sys
from argparse import ArgumentParser, BooleanOptionalAction
from model import Game
from pyqtgraph.Qt import QtGui

# define arguments
parser = ArgumentParser()
parser.add_argument("-m", "--mapping",
                    action=BooleanOptionalAction,
                    default=False,
                    help="Starts the mapping of the board")

parser.add_argument("-st", "--start",
                    action=BooleanOptionalAction,
                    default=False,
                    help="Chess game starts")

parser.add_argument("-t", "--train",
                    action=BooleanOptionalAction,
                    default=False,
                    help="Photograph a square of the board for training")

parser.add_argument("-sn", "--snap",
                    action=BooleanOptionalAction,
                    default=False,
                    help="Snaps a photo of the board")

args = vars(parser.parse_args())

if __name__ == "__main__":
  # calibration mapping
  if args['mapping']:
    app = QtGui.QApplication(sys.argv)
    game = Game()
    game.mapping()

  # start a game
  if args['start']:
    app = QtGui.QApplication(sys.argv)
    game = Game()
    game.start()
    sys.exit(app.exec_())

  # take pictures for training
  if args['train']:
    app = QtGui.QApplication(sys.argv)
    game = Game()
    game.train()
    sys.exit(app.exec_())

  # take pictures for training
  if args['snap']:
    app = QtGui.QApplication(sys.argv)
    game = Game()
    game.snap()
    sys.exit(app.exec_())