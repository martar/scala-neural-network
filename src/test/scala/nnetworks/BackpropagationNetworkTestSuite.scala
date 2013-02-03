package nnetworks

import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.FunSuite
import nnetworks.Network
import nnetworks.Layer
import nnetworks.Generator
import nnetworks.Functions

/**
 * This class is a test suite for the methods in object FunSets. To run
 * the test suite, you can either:
 *  - run the "test" command in the SBT console
 *  - right-click the file in eclipse and chose "Run As" - "JUnit Test"
 */
@RunWith(classOf[JUnitRunner])
class BackpropagationNetworkTestSuite extends FunSuite {

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
    val weights: List[List[Double]] = List(List(-30., 20, 20))
    val layer = new Layer(Functions.steep(0.5), weights)

    assert(layer.eval(List(1., 0., 0.)) === List(0))
    assert(layer.eval(List(1., 0., 1.)) === List(0))
    assert(layer.eval(List(1., 1., 0.)) === List(0))
    assert(layer.eval(List(1., 1., 1.)) === List(1))

  }

  test("Test XOR first layer network implemented") {
    val weights: List[List[Double]] = List(List(-30., 20., 20.), List(10., -20., -10.))
    val layer = new Layer(Functions.steep(0.5), weights)

    assert(layer.eval(List(1., 0., 0.)) === List(0, 1))
    assert(layer.eval(List(1., 0., 1.)) === List(0, 0))
    assert(layer.eval(List(1., 1., 0.)) === List(0, 0))
    assert(layer.eval(List(1., 1., 1.)) === List(1, 0))

  }

  test("Test AND network implemented with activation function - Network! with explicit Bias") {
    val weights: List[List[Double]] = List(List(-30.0, 20, 20))
    val layer = new Layer(Functions.steep(0.5), weights)
    val net = new Network(List(layer), true)
    assert(net.eval(List(0., 0.)) === List(0))
    assert(net.eval(List(0., 1.)) === List(0))
    assert(net.eval(List(1., 0.)) === List(0))
    assert(net.eval(List(1., 1.)) === List(1))
  }

  test("Test AND network implemented lab") {
    val weights: List[List[Double]] = List(List(0.34, -0.14, -0.97))
    val layer = new Layer(Functions.identitiy, weights)
    val net = new Network(List(layer), true)
    println(net.eval(List(0., 0.)))
    println(net.eval(List(0., 1.)))
    println(net.eval(List(1., 0.)))
    println(net.eval(List(1., 1.)))
    //assert(net.eval(List(0., 0.)) === List(0))
    //assert(net.eval(List(0., 1.)) === List(0))
    //assert(net.eval(List(1., 0.)) === List(0))
    //assert(net.eval(List(1., 1.)) === List(1))
  }

  test("Test AND network implemented lab2") {
    val weights: List[List[Double]] = List(List(-8.93, 5.89, 5.89))
    val layer = new Layer(Functions.sigmoid, weights)
    val net = new Network(List(layer), true)
    println(net.eval(List(0.7, 0.7)))
    //println(net.eval(List(0., 1.)))
    //println(net.eval(List(1., 0.)))
    //println(net.eval(List(1., 1.)))
    //assert(net.eval(List(0., 0.)) === List(0))
    //assert(net.eval(List(0., 1.)) === List(0))
    //assert(net.eval(List(1., 0.)) === List(0))
    //assert(net.eval(List(1., 1.)) === List(1))
  }

  test("Test XOR with steep activation function") {
    val activation = Functions.steep(0.5)
    val weights: List[List[Double]] = List(List(-30.0, 20, 20), List(10., -20., -10.))
    val layer = new Layer(activation, weights)

    val weights2: List[List[Double]] = List(List(-10.0, 20, 20))
    val layer2 = new Layer(activation, weights2)

    val net = new Network(List(layer, layer2), true)
    assert(net.eval(List(0., 0.)) === List(1))
    assert(net.eval(List(0., 1.)) === List(0))
    assert(net.eval(List(1., 0.)) === List(0))
    assert(net.eval(List(1., 1.)) === List(1))
  }

  test("Random weights generator") {
    assert(Generator.randomWeigth(true)(0, 1)(2, 2).length === 2)
  }

  test("Network with random weights") {
    val layer = new Layer(Functions.steep(0.5), Generator.randomWeigth(true)(-30, 30), 2, 2)
    val net = new Network(List(layer), true)
    assert((layer.weights.length === 2))
    // 3  because we expect the layer to generate weight to bias neuron as well
    assert((layer.weights(0).length === 3))
  }

  ignore("Tets reading from file") {
    val input = Generator.getInputsFromFile("E:/Projects/scala-neural-network/src/test/scala/resources/test_input.txt")
    val weights: List[List[Double]] = List(List(-30.0, 20, 20))
    val layer = new Layer(Functions.steep(0.5), weights)
    val net = new Network(List(layer), true)
    assert(net.eval(input) === List(0))
    
  }

   
  test("Test Backpropagation") {
    val trainingSet = List(List(0., 0.), List(0., 1.), List(1., 0.), List(1., 1.))
    val teacherSet = List(List(0.), List(1.), List(1.), List(1.))
    val weights: List[List[Double]] = List(List(math.random, math.random,math.random), List(math.random, math.random, math.random))
    val layer = new Layer(Functions.sigmoid, weights)

    val weights2: List[List[Double]] = List(List(math.random, math.random, math.random))
    val layer2 = new Layer(Functions.sigmoid, weights2)
    
    val net = new Network(List(layer, layer2), true)
    val alfa = 0.3
    println("BEFORE LERNING:")
    println(net.layers)
    println("LERNING:")
    net.backPropagaton((trainingSet, teacherSet).zipped.toList, alfa)
    println("WEIGHTS AFTER LEARNIGN:")
    println(net.layers)
  }

    
   
}
