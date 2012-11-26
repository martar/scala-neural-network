package nnetworks

import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.FunSuite

/**
 * This class is a test suite for the methods in object FunSets. To run
 * the test suite, you can either:
 *  - run the "test" command in the SBT console
 *  - right-click the file in eclipse and chose "Run As" - "JUnit Test"
 */
@RunWith(classOf[JUnitRunner])
class KohonenNetworkTestSuite extends FunSuite {

  test("Test always two layers in Network") {

    val input1 = List(
      1., 0., 0.,
      1, 0., 0,
      1, 0, 0)
    val input2 = List(
      0., 0., 1.,
      0, 0., 1,
      0, 0, 1)
    val input3 = List(
      0., 0., 0.,
      0, 0., 0,
      0, 0, 0)
    val input4 = List(
      1., 1., 1.,
      1, 1., 1,
      1, 1, 1)

    val trainingSet = List(input1, input2, input3, input4)

    val test1 = List(
      1., 1., 1.,
      1, 0., 1,
      1, 1, 1)
    val test2 = List(
      1., 0., 0.,
      1, 1., 0,
      1, 0, 0)
    val test3 = List(
      0., 0., 1.,
      0, 1., 1,
      0, 0, 1)
    val layer = new Layer(Functions.identitiy, Generator.randomWeigth(false)(0, 0.1), trainingSet.length, trainingSet(0).length)
    val net = new KohonenNetwork(List(layer))
    // traning et, iterations, neighbour_range, conscience, alfa, beta
    net.learn(trainingSet, 32000, 1, 0.75, 0.03, 0.25)
    println(net.getWinner(input1))
    println(net.getWinner(input2))
    println(net.getWinner(input3))
    println(net.getWinner(input4))
    println("--------------")
    println(net.getWinner(test1))
    println(net.getWinner(test2))
    println(net.getWinner(test3))
  }

}


