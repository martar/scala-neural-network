package nnetworks

object Functions {
  def roundedSigmoid(v: Double): Double = math.round(1.0 / (1.0 + math.exp(-v)))
}

/**
 * layer is a two dimensional array of doubles - weights
 */
class Layer(activation: (Double) => Double, weights: Array[Array[Double]]) {

  /**
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
    res
  }

  def eval(input: Array[Double]): Array[Double] = {
    multiThreadedIdiomatic(input, weights) map (activation)
  }
}


class Network(layers: List[Layer]) {
  def add(l: Layer) = new Network(l :: layers)
  def eval(input: Array[Double]): Array[Double] = {
    def eval0(layers: List[Layer], input: Array[Double]): Array[Double] = layers match {
      case Nil => input
      case head :: tail => eval0(tail, head.eval(input))
    }
    eval0(layers, input)
  }
}