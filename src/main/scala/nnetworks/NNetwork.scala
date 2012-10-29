package nnetworks

import scala.Array.canBuildFrom
import scala.util.Random

object Functions {
  def sigmoid(v: Double): Double = 1.0 / (1.0 + math.exp(-v))
  /**
   * return 0 for values below and equal to boundary, and 1 for values greater that boundary
   */
  def steep(boundary: Double): Double => Double = {
    def steep_0(v: Double): Double = if (v > boundary) 1 else 0
    steep_0
  }
  def identitiy(v: Double): Double = v

}

object Generator {

  def randomWeigth(min: Double, max: Double)(currLayerSize: Int, nextLayerSize: Int): List[List[Double]] = {
    def randomWithinRange(min: Double, max: Double) = min + (max - min) * Random.nextDouble()
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

class Layer(activation: Double => Double, weights: List[List[Double]]) {

  def getWeights: List[List[Double]] = weights

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
}

class Network(layers: List[Layer]) {
  def eval(input: List[Double]): List[Double] = {
    def eval0(layers: List[Layer], input: List[Double]): List[Double] = layers match {
      case Nil => input.tail // we return tail, because 'head' is a bias neuron, that doesn't matter on final output
      case head :: tail => eval0(tail, 1 :: head.eval(input)) //we add a bias neuron set to 1

    }
    //add a bias neuron set to 1
    eval0(layers, 1 :: input)
  }

}