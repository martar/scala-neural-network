package nnetworks

object Functions {
  def roundedSigmoid(v: Double): Double = math.round(1.0 / (1.0 + math.exp(-v)))
}

/**
 * layer is a two dimensional array of doubles - weights
 */
class Layer(weights: Array[Array[Double]]) {

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

  def applyInputs(input: Array[Double], fun: Double => Double): Array[Double] = {
    multiThreadedIdiomatic(input, weights) map (fun(_))
  }

  def applyInputs(input: Array[Double]): Array[Double] = {
    multiThreadedIdiomatic(input, weights)
  }

}
