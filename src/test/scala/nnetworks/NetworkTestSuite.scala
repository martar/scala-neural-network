package nnetworks

import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.FunSuite
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner

/**
 * This class is a test suite for the methods in object FunSets. To run
 * the test suite, you can either:
 *  - run the "test" command in the SBT console
 *  - right-click the file in eclipse and chose "Run As" - "JUnit Test"
 */
@RunWith(classOf[JUnitRunner])
class NetworkTestSuite extends FunSuite {

  test("Test sigmoid funciton") {
    assert(Functions.steep(0.5)(Functions.sigmoid(-4)) === 0)
    assert(Functions.steep(0.5)(Functions.sigmoid(0)) === 0)
    assert(Functions.steep(0.5)(Functions.sigmoid(4)) === 1)
  }
  test("Test steep activation funciton") {
    assert(Functions.steep(0.5)(0.5) === 0)
    assert(Functions.steep(0.5)(0.2) === 0)
    assert(Functions.steep(0.5)(0.6) === 1)
  }

  test("Test AND layer implemented with activation function") {
    val weights: Array[Array[Double]] = Array(Array(-30.0), Array(20), Array(20))
    val layer = new Layer(Functions.steep(0.5), weights)

    assert(layer.eval(List(1., 0., 0.)) === List(0))
    assert(layer.eval(List(1., 0., 1.)) === List(0))
    assert(layer.eval(List(1., 1., 0.)) === List(0))
    assert(layer.eval(List(1., 1., 1.)) === List(1))

  }

  test("Test XOR first layer network implemented") {
    val weights: Array[Array[Double]] = Array(Array(-30.0, 10.), Array(20, -20.), Array(20, -10.))
    val layer = new Layer(Functions.steep(0.5), weights)

    assert(layer.eval(List(1., 0., 0.)) === List(0, 1))
    assert(layer.eval(List(1., 0., 1.)) === List(0, 0))
    assert(layer.eval(List(1., 1., 0.)) === List(0, 0))
    assert(layer.eval(List(1., 1., 1.)) === List(1, 0))

  }

  test("Test AND network implemented with activation function - Network! with explicit Bias") {
    val weights: Array[Array[Double]] = Array(Array(-30.0), Array(20), Array(20))
    val layer = new Layer(Functions.steep(0.5), weights)
    val net = new Network(List(layer))
    assert(net.eval(List(0., 0.)) === List(0))
    assert(net.eval(List(0., 1.)) === List(0))
    assert(net.eval(List(1., 0.)) === List(0))
    assert(net.eval(List(1., 1.)) === List(1))
  }

  test("Test XOR network implemented with steep activation function - Network!") {
    val activation = Functions.steep(0.5)
    val weights: Array[Array[Double]] = Array(Array(-30.0, 10.), Array(20, -20.), Array(20, -10.))
    val layer = new Layer(activation, weights)

    val weights2: Array[Array[Double]] = Array(Array(-10.0), Array(20), Array(20))
    val layer2 = new Layer(activation, weights2)

    val net = new Network(List(layer, layer2))
    assert(net.eval(List( 0., 0.)) === List(1))
    assert(net.eval(List( 0., 1.)) === List(0))
    assert(net.eval(List(1., 0.)) === List(0))
    assert(net.eval(List(1., 1.)) === List(1))
  }

}
