
import weka.core.*;
import weka.core.neighboursearch.KDTree;

import java.util.ArrayList;

public class WekaKdTreeList {
    public static void main(String[] args) throws Exception {

        //--------------Cast hodnot
        ArrayList<Attribute> attr = new ArrayList<>(3);


        //--------------Cast atributov
        Attribute vek = new Attribute("vek", 0);
        Attribute cielovaTrieda = new Attribute("cielovaTrieda", 2);

        attr.add(vek);
        attr.add(cielovaTrieda);


        Instances baseInstances = new Instances("Test", attr, 2);
        baseInstances.setClassIndex(baseInstances.numAttributes() - 1);

        System.out.println("Before adding any instance");
        System.out.println("--------------------------");
        System.out.println(baseInstances);
        System.out.println("--------------------------");

        int size = baseInstances.numAttributes();
        int pocet = 5;

        //plnenie instancii
        for (int i = 0; i < pocet; i++) {
            double[] instanceValue = new double[size];

            instanceValue[0] = i * 10;
            instanceValue[1] = (i + 1) % 2;

            baseInstances.add(new DenseInstance(1d, instanceValue));
        }
        System.out.println("After adding any instance");
        System.out.println("--------------------------");
        System.out.println(baseInstances);
        System.out.println("--------------------------");

        KDTree knn = new KDTree();
        knn.setInstances(baseInstances);
        double[] newValues = new double[size];
        newValues[0] = 10;
        newValues[1] = 1;
        Instances newInstances = new Instances(
                "Target",
                attr,
                2
        );

        newInstances.add(new DenseInstance(1d, newValues));

        Instances nearestInstances = knn.kNearestNeighbours(newInstances.firstInstance(), 2);


        for (int i = 0; i < nearestInstances.numInstances(); i++) {
            System.out.println(nearestInstances.instance(i).value(vek) + ", "
                    + nearestInstances.instance(i).value(cielovaTrieda));
        }



    }
}
