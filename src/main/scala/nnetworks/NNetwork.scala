package nnetworks

import scala.Array.canBuildFrom

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

/**
 * layer is a two dimensional array of doubles - weights
 */
class Layer(activation: Double => Double, weights: Array[Array[Double]]) {

  /**
   * Multiply input vector with weight matrix
   * Fastest implementation based on http://blog.scala4java.com/2011/12/matrix-multiplication-in-scala-single.html
   */
  def multiThreadedIdiomatic(m1: Array[Double], m2: Array[Array[Double]]) = {
    val res = Array.fill[Double](m2(0).length)(0.0)
    for (
      col <- (0 until m2(0).length).par;
      i <- 0 until m1.par.length
    ) {
      res(col) += m1(i) * m2(i)(col)
    }
    res.toList
  }

  /**
   * Evaluate the layer: multiply input with weights and apply activation function to each neuron
   */
  def eval(input: List[Double]): List[Double] = {
    multiThreadedIdiomatic(input.toArray, weights) map (activation)
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