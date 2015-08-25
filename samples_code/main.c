/* First: disable MCMC flag to force the structure to read full halo_ids_data structure */
#undef MCMC

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/stat.h>
#include "../code/allvars.h"
#include "../code/proto.h"
#include <sqlite3.h> 

void load_tree_table(int sim, int filenr)
{
  int i,j, n, totNHalos, SnapShotInFileName;
  char buf[1000];
  SnapShotInFileName=LastDarkMatterSnapShot;


#ifndef MRII
  sprintf(buf, "%s/treedata/tree_dbids_%03d.%d", SimulationDir, SnapShotInFileName, filenr);
#else
  sprintf(buf, "%s/treedata/tree_sf1_dbids_%03d.%d", SimulationDir, SnapShotInFileName, filenr);
#endif
  if(!(treedbids_file = fopen(buf, "r")))
    {
      char sbuf[1000];
      sprintf(sbuf, "can't open file `%s'\n", buf);
      terminate(sbuf);
    }



#ifndef MRII
  sprintf(buf, "%s/treedata/trees_%03d.%d", SimulationDir, SnapShotInFileName, filenr);
#else
  sprintf(buf, "%s/treedata/trees_sf1_%03d.%d", SimulationDir, SnapShotInFileName, filenr);
#endif

  if(!(tree_file = fopen(buf, "r")))
    {
      char sbuf[1000];
      sprintf(sbuf, "can't open file place `%s'\n", buf);
      terminate(sbuf);
    }

  //read header on trees_** file
  myfread(&Ntrees, 1, sizeof(int), tree_file);
  myfread(&totNHalos, 1, sizeof(int), tree_file);

  TreeNHalos = mymalloc("TreeNHalos", sizeof(int) * Ntrees);
  myfread(TreeNHalos, Ntrees, sizeof(int), tree_file);

  Halo_Data = mymalloc("Halo_Data", sizeof(struct halo_data) * totNHalos);
  myfseek(tree_file, sizeof(int) * (2 + Ntrees), SEEK_SET);
  myfread(Halo_Data, totNHalos, sizeof(struct halo_data), tree_file);


  HaloIDs_Data = mymalloc("HaloIDs_Data", sizeof(struct halo_ids_data) * totNHalos);
  myfseek(treedbids_file, 0, SEEK_SET);
  myfread(HaloIDs_Data, totNHalos, sizeof(struct halo_ids_data), treedbids_file);


}

/**@file main.c
   Only use for single simulation (22/08/2015).
   Use SQLite to simplify things.
/**@brief Main routine to generate sample files for MCMC process*/

int main(int argc, char **argv) {
  int i,j,k;
  sqlite3 *db;
  char *zErrMsg = 0;
  int rc;
  char *sql;
  struct halo_data halo;
  struct halo_ids_data halo_ids;
  if(argc > 3) {
      printf("\n  usage: gen_sample <parameterfile>\n\n");
      endrun(0);
    }

  /*Reads the parameter file, given as an argument at run time. */
  read_parameter_file(argv[1]);
  
  /* Open database in memory */
  rc = sqlite3_open(":memory:", &db);
  if( rc ){
    fprintf(stderr, "Can't open database: %s\n", sqlite3_errmsg(db));
    exit(0);
  }else{
    fprintf(stderr, "Opened database successfully\n");
  }
  /* Read in trees and store in sqlite */
  

  /* close db and finish the job */
  sqlite3_close(db);
  return 0;
}
