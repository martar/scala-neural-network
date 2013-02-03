package nnetworks

import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.FunSuite
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
class KohonenGrossbergNetworkTestSuite extends FunSuite {

  test("Test Kohonen Grossberg") {

    val input1 = List(
      1., 1., 1.,
      0., 0., 0.,
      0., 0., 0.)
    val input2 = List(
      0., 0., 0.,
      1, 1., 1,
      0, 0, 0)
    val input3 = List(
      0., 0., 0.,
      0, 0., 0,
      1, 1, 1)
    val input4 = List(
      1., 0., 0.,
      1, 0., 0,
      1, 0, 0)     
    val input5 = List(
      0., 1., 0.,
      0., 1., 0.,
      0., 1, 0.)
     val input6 = List(
      0., 0., 1.,
      0, 0., 1,
      0, 0, 1)
    val input7 = List(
      1., 0., 0.,
      0, 0., 0,
      0, 0, 0)
    val input8 = List(
      0., 0., 0.,
      0, 0., 0,
      0, 0, 1)
    val input9 = List(
      0., 0., 1.,
      0., 1., 0.,
      1., 0., 0.)
      
    val output1 = List(0., 0., 1.)
    val output2 = List(0., 0., 1.)
    val output3 = List(0., 0., 1.)
    val output4 = List(0., 1., 0.)    
    val output5 = List(0., 1., 0.)
    val output6 = List(0., 1., 0.)
    val output7 = List(1., 0., 0.)
    val output8 = List(1., 0., 0.)
    val output9 = List(1., 0., 0.)
      
    val trainingSet = List(input1, input2, input3, input4
        ,input5, input6, input7, input8, input9)

    val outputsSet = List(output1, output2, output3, output4, output5, output6, output7, output8, output9)
        
    val test1 = List(
      0., 0., 0.,
      0, 1., 0,
      1, 0, 0)
    val test2 = List(
      0., 0., 0.,
      1, 0., 0,
      1, 0, 0)
    val test3 = List(
      0., 0., 0.,
      0, 1., 1,
      0, 0, 0)
          
    val klayer = new Layer(Functions.identitiy, Generator.randomWeigth(false)(0, 0.1), trainingSet.length, trainingSet(0).length)
    val glayer = new Layer(Functions.identitiy, Generator.randomWeigth(false)(0, 0.1), 3, 9)
    /*
    val net = new KohonenGrossbergNetwork(List(klayer,glayer),false)
    // traning et, iterations, neighbour_range, conscience, alfa, beta
    net.learn(trainingSet, 20000, 1, 0.75, 0.06, 0.25, outputsSet)
    
    net.getGrossRes(input1)
    net.getGrossRes(input2)
    net.getGrossRes(input3)    
    println("--------------")
    net.getGrossRes(test1)
    net.getGrossRes(test2)
    net.getGrossRes(test3)    
    */
    
  }
}


