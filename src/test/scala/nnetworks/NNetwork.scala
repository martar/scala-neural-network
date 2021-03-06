package nnetworks
import scala.util.Random

object Functions {
  def sigmoid(v: Double): Double = 1.0 / (1.0 + math.exp(-v))

  def derivSigmoid(v: Double) = math.exp(v) / math.pow((math.exp(v) + 1), 2)
  /**
   * return 0 for values below and equal to boundary, and 1 for values greater that boundary
   */
  def steep(boundary: Double): Double => Double = {
    def steep_0(v: Double): Double = if (v > boundary) 1 else 0
    steep_0
  }
  def identitiy(v: Double): Double = v
  def dIdentity(v: Double) = 1.
}

object Generator {

  def randomWeigth(bias: Boolean)(min: Double, max: Double)(currLayerSize: Int, nextLayerSize: Int): List[List[Double]] = {
    def trunc(x: Double) = math.round(x * 100) * 0.01
    def randomWithinRange(min: Double, max: Double) = trunc(min + (max - min) * Random.nextDouble())

    def randomWeight_0(currLayerSize: Int, nextLayerSize: Int, accu: List[List[Double]]): List[List[Double]] = currLayerSize match {
      case 0 => accu
      case _ => randomWeight_0(currLayerSize - 1, nextLayerSize, List.fill(nextLayerSize)(randomWithinRange(min, max)) :: accu)
    }
    randomWeight_0(currLayerSize, nextLayerSize, List())

  }

  def getInputsFromFile(path: String): List[Double] = {
    def parseDouble(s: String): Double = try { s.toDouble } catch { case _ => 0. }

    io.Source.fromFile(path).getLines().toList map parseDouble _
  }
}

/**
 * layer is a two dimensional array of doubles - weights
 */

class Layer(activation: Double => Double, var weights: List[List[Double]]) {

  // old weight updateds for backpropagation momentum
  var previousWeightUpdates: List[List[Double]] = (for (neuron <- weights) yield {
    (for (weight <- neuron) yield 0.0).toList
  }).toList

  def setWeights(newWeights: List[List[Double]]) = {
    weights = newWeights
  }

  def this(activation: Double => Double, weightGen: (Int, Int) => List[List[Double]], currLayerSize: Int, nextLayerSize: Int) =
    // adding one because of the Bias neuron
    this(activation: Double => Double, weightGen(currLayerSize, nextLayerSize + 1))

  def multiThreadedLinearCombination(input: List[Double], matrix: List[List[Double]]) = {

    def helper(input: List[Double], matrix: List[List[Double]], accu: List[Double]): List[Double] = matrix match {
      case Nil => accu.reverse
      case head :: tail => helper(input, tail, (head.par.zip(input.par) map ((tuple: (Double, Double)) => tuple._1 * tuple._2) sum) :: accu)
    }
    helper(input, matrix, List())

  }

  /**
   * Evaluate the layer: multiply input with weights and apply activation function to each neuron
   */
  def eval(input: List[Double]): List[Double] = {
    multiThreadedLinearCombination(input, weights) map (activation)
  }

  /**
   * Evaluate the layer with custom activation multiply input with weights and apply activation function to each neuron
   */
  def eval(input: List[Double], activationFun: Double => Double): List[Double] = {
    multiThreadedLinearCombination(input, weights) map (activationFun)
  }

  override def toString = weights.toString()
}

/**
 * Neural Network. bias - true if the Network layers should contain bias neuron and weights for bias are
 * provided in layers
 */
class Network(var layers: List[Layer], bias: Boolean) {

  def getLayers = layers

  def getFirstLayer = layers.head

  var deltas: List[List[Double]] = List()
  var evaluatedDeriv: List[List[Double]] = List()
  var ys: List[List[Double]] = List()

  def setLayers(newlayers: List[Layer]) = {
    layers = newlayers
  }

  def eval(input: List[Double]): List[Double] = {
    val biasValue = if (bias) 1 else 0
    def eval0(layers: List[Layer], input: List[Double]): List[Double] = layers match {
      case Nil => input.tail // we return tail, because 'head' is a bias neuron, that doesn't matter on final output
      case head :: tail =>
        val evaled: List[Double] = head.eval(input)
        val res: List[Double] = biasValue :: evaled
        ys = res :: ys
        eval0(tail, res) //we add a bias neuron set to 1

    }
    //add a bias neuron set to 1
    eval0(layers, biasValue :: input)
  }

  def eval(input: List[Double], activationFun: Double => Double): List[Double] = {
    val biasValue = if (bias) 1 else 0
    def eval0(layers: List[Layer], input: List[Double]): List[Double] = layers match {
      case Nil => input.tail // we return tail, because 'head' is a bias neuron, that doesn't matter on final output
      case head :: tail =>
        val evaled: List[Double] = head.eval(input, activationFun)
        val res: List[Double] = evaled
        evaluatedDeriv = res :: evaluatedDeriv
        eval0(tail, res) //we add a bias neuron set to 1

    }
    //add a bias neuron set to 1
    eval0(layers, biasValue :: input)
  }

  def computeDistance(weights: List[Double], inputValues: List[Double]) = {
    inputValues.zip(weights).map((tuple: (Double, Double)) => math.pow(tuple._1 - tuple._2, 2)) sum
  }

  def computeErrors(weights: List[Double], inputValues: List[Double]) = {
    inputValues.zip(weights).map((tuple: (Double, Double)) => math.pow(tuple._1 - tuple._2, 2))
  }

  def computeDeltas(previousLayer: List[List[Double]], currentDeltas: List[Double]) = {

    def w1(input: List[(List[Double], Double)]) = {

      def w0(curr: List[(List[Double], Double)], accu: List[List[Double]]): List[List[Double]] = curr match {
        case Nil =>
          accu
        case head :: tail => head match {
          case (list, delta) =>
            w0(tail, list.map((weight: Double) => weight * delta) :: accu)

        }

      }
      w0(input, List())
    }

    val result = w1((previousLayer, currentDeltas).zipped.toList)

    def sum(a: (Double, Double)) = a match { case (first, second) => first + second }

    def folding(a: List[Double], b: List[Double]) = (a, b).zipped.toList.map(sum)

    result.reduceLeft(folding)

  }

  def backProp(netLayers: List[Layer], errors: List[Double]) {

    def backProp0(netLayers: List[Layer], currentDeltas: List[Double], deltasAccu: List[List[Double]]): List[List[Double]] = netLayers match {
      case Nil => deltasAccu
      case head :: tail =>
        val lol = computeDeltas(head.weights, currentDeltas).tail
        backProp0(tail, lol, lol :: deltasAccu)
    }

    deltas = backProp0(netLayers, errors, List(errors)).tail
  }

  def backPropagaton(learningSet: List[(List[Double], List[Double])], alfa: Double) {

    var step = 0;
    var gerror = 100.0;
    while (gerror > 0.25 && step < 50000) {
      step += 1
      var error = 0.0;
      learningSet.foreach((item) => item match {
        case (input, expectedOutput) => {
          deltas = List()
          evaluatedDeriv = List()
          ys = List()
          this.eval(input, Functions.derivSigmoid)
          val evaluatedNet = this.eval(input)
          val errors = computeErrors(evaluatedNet, expectedOutput)
          error += math.sqrt(errors.sum) / errors.length
          val prev = layers.head

          // compute deltas and sote it in the class variable
          backProp(layers.reverse, errors)
          evaluatedDeriv = evaluatedDeriv.reverse
          val lol: List[Double] = 1 :: input
          ys = lol :: (ys.tail).reverse
          /**
           * print("DELTAS:")
           * println(deltas)
           * print("EVALUATED DERIV:")
           * println(evaluatedDeriv)
           * print("YS:")
           * println(ys)
           * print("LAYERS")
           * println(this.layers)
           * println("---------------------")
           */
          for (i <- 0 to this.layers.length - 1) {
            var newWeights: List[List[Double]] = List()
            val layer = this.layers(i)
            val delta = deltas(i)
            val derivs = evaluatedDeriv(i)
            val y = ys(i)
            for (j <- 0 to layer.weights.length - 1) {
              val mapped = layer.weights(j).map((w: Double) => w + alfa * delta(j) * derivs(j))
              newWeights = mapped :: newWeights
            }
            this.layers(i).weights = newWeights
          }
        }
      })
      gerror = error / learningSet.length
      print("ERROR:")
      println(gerror)
      gerror
    }
  }

}

class KohonenNetwork(layers: List[Layer]) extends Network(layers, false) {

  val len = getFirstLayer.weights(0).length
  var conscience = List.fill(len - 1)(1.0 / len)

  def getDistancesVector(allNeuronsWeights: List[List[Double]], inputValues: List[Double], accu: List[Double]): List[Double] = allNeuronsWeights match {
    case Nil => accu
    case head :: tail => getDistancesVector(tail, inputValues, computeDistance(head, inputValues) :: accu)
  }

  def getWinner(inputValues: List[Double]) = {
    getDistancesVector(getFirstLayer.weights, inputValues, List()).zipWithIndex.min._2
  }

  def getWinnerWhileLearning(inputValues: List[Double], cons: Double, beta: Double) = {

    def apply_conscience(distances: List[Double], conscienceVector: List[Double]) = {
      distances.zip(conscienceVector).map((tuple: (Double, Double)) => tuple._1 + cons * (distances.length * tuple._2 - 1.0))
    }

    def update_conscience(winnerId: Int, beta: Double) {
      conscience.zipWithIndex.map((tuple: (Double, Int)) => if (tuple._2 == winnerId) tuple._1 - beta * (1.0 - tuple._1) else tuple._1 + beta * (0.0 - tuple._1))

    }
    val distances = apply_conscience(getDistancesVector(getFirstLayer.weights, inputValues, List()), conscience)
    val winnerId = distances.zipWithIndex.min._2
    update_conscience(winnerId, beta)
    winnerId

  }

  def gen_result(inputValues: List[Double]) = {
    val winner = getWinner(inputValues)
    List.fill(getFirstLayer.weights.length)(0).zipWithIndex.map((tuple: (Int, Int)) => if (tuple._2 == winner) 1 else 0)
  }

  /*
   * alfa jest sta�� uczenia, wybieran� na og� pomi�dzy 0,1 i 0,7. 
   * Dzi�ki temu lekko "popychamy" wektor wag w kierunku wektora danych. 
   * Algorytm ten wykazuje niekiedy rosn�c� niestabilno��, dlatego cz�sto stosuje si� obecnie metod� subtraktywn�. 
   * Opiera si� ona na spostrze�eniu, �e pewnym sposobem lekkiego przyci�gania 
   * wektora wag w kierunku wektora wej�ciowego jest odj�� je, a nast�pnie cz�� tej r�nicy doda� do wektora wag
   */
  def learn(trainingSet: List[List[Double]], steps: Int, neighbour_range: Int, cons: Double, alfa: Double, beta: Double) = {
    val len = trainingSet(0).length

    def learn_step(inputValues: List[Double], alfa: Double, beta: Double) = {

      def gen_new_weight() = {
        def get_neigbours_indexes(neuronIndex: Int, dist: Int) = {
          (0 to getFirstLayer.weights.length) filter { i => math.abs(neuronIndex - i) <= dist }
        }
        //to do 2D

        val neighgour_indexes = get_neigbours_indexes(getWinnerWhileLearning(inputValues, cons, beta), neighbour_range)

        def gen_new_weight0(list: List[List[Double]], index: Int, accu: List[List[Double]]): List[List[Double]] = (list, neighgour_indexes contains index) match {
          case (Nil, _) => accu.reverse
          case (head :: tail, false) => gen_new_weight0(tail, index + 1, head :: accu)
          case (head :: tail, true) => {
            val new_weights = head.map { (weight: Double) => weight + alfa * math.exp(-math.pow(getWinner(inputValues) - index, 2) / math.pow(neighbour_range, 2)) * (inputValues(index) - weight) }
            gen_new_weight0(tail, index + 1, new_weights :: accu)
          }
        }

        gen_new_weight0(getFirstLayer.weights, 0, List())
      }

      getFirstLayer.setWeights(gen_new_weight())

    }

    for (i <- 1 to steps) {
      for (j <- 0 to trainingSet.length - 1) {

        learn_step(trainingSet(j), alfa, beta)
      }

    }

  }

}