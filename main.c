#include <stdio.h>
#include <stdlib.h>
#include <string.h>


/* structures */
typedef struct student {
    char *name;
    int  id;
    struct clist *courses;
} student;

typedef struct course {
    char *title;
    int  number;
    struct slist *students;
} course;

typedef struct slist {
    student      *info;
    struct slist *next;
} slist;

typedef struct clist {
    course       *info;
    struct clist *next;
} clist;


/* prototypes */
slist* add_student(slist *students, char *name, int id);
clist* add_course(clist *courses, char *title, int number);
void reg_student(slist *students, clist *courses, int id, int number);
void unreg_student(slist *students, int id, int number);
void print_students(slist *students);
void print_courses(clist *courses);
void free_all(slist *sl, clist *cl);

/*my functions*/
slist* find_student(slist *students,int id);
clist* find_course(clist *corses,int number);
void insert_course(clist *courses,int number,clist *course,slist *student);
void insert_student(slist *students,int id ,slist *student,clist *course);
void remove_course(slist *student,int number,clist *course);
void remove_student(clist *course,int id,slist *student);
void free_student(student *student);
void free_course(course *course);

static void getstring(char *buf, int length) {
    int len;
    buf = fgets(buf, length, stdin);
    len = (int) strlen(buf);
    if (buf[len-1] == '\n')
        buf[len-1] = '\0';
}

int main() {
    slist* students = 0;
    clist* courses = 0;
    char  c;
    char  buf[100];
    int   id, num;

    do {
        printf("Choose:\n"
               "    add (s)tudent\n"
               "    add (c)ourse\n"
               "    (r)egister student\n"
               "    (u)nregister student\n"
               "    (p)rint lists\n"
               "    (q)uit\n");

        while ((c = (char) getchar()) == '\n');
        getchar();

        switch (c) {
            case 's':
                printf("Adding new student.\n");

                printf("Student name: ");
                getstring(buf, 100);

                printf("Student ID: ");
                scanf("%d", &id);

                students = add_student(students, buf, id);

                break;

            case 'c':
                printf("Adding new course.\n");

                printf("Course name: ");
                getstring(buf, 100);

                printf("Course number: ");
                scanf("%d", &num);

                courses = add_course(courses, buf, num);

                break;

            case 'r':
                printf("Registering a student to a course.\n");

                printf("Student ID: ");
                scanf("%d", &id);

                printf("Course number: ");
                scanf("%d", &num);

                reg_student(students, courses, id, num);

                break;

            case 'u':
                printf("Unregistering a student from a course.\n");

                printf("Student ID: ");
                scanf("%d", &id);

                printf("Course number: ");
                scanf("%d", &num);

                unreg_student(students, id, num);

                break;

            case 'p':
                printf("Printing Information.\n");

                print_students(students);
                print_courses(courses);

                break;

            case 'q':
                printf("Quitting...\n");
                break;
        }

        if (c != 'q')
            printf("\n");
    } while (c != 'q');

    free_all(students,courses);
    return 0;
}

/*functions*/
slist* add_student(slist *students, char *name, int id) {
    slist *cur_student ;
    slist *new_student;
    slist *perv_student;
    student *std =  (struct student*)malloc(sizeof(struct student)); /*create pointer to a new student*/
    if(std==NULL){
        exit(1);
    }
    /*insert std info*/
    std->id = id;
    /*allocate memory for std->name*/
    std->name=(char*) malloc((strlen(name)+1)*sizeof(char));/*have to allocate, unless it is a pointer*/
    if(std->name==NULL){
        exit(1);
    }
    strcpy(std->name,name);
    std->courses=0;/*initial the course list to NULL*/
    /*insert student to the student list*/
    /*temp changing pointer student*/
    
    cur_student = students;
    /*create a new node in the student list*/
    new_student =(struct slist*) malloc((sizeof (struct slist)));
    if(new_student==NULL){
        exit(1);
    }
    new_student->info =std;/*initial the info pointer of the student*/
    if(!students){/*in case that the list is empty*/
        new_student->next=0;
        return new_student;
    }
    if(std->id < students->info->id){/*in case that the id of the head of the list is bigger than the new student*/
        new_student->next=students;
        return new_student;
    }
    
    while (cur_student){/*in case need to insert the student in middle location or at the end*/
        if(std->id > cur_student->info->id) {
            perv_student = cur_student;
            cur_student = cur_student->next;
        }
        else{
            break;
        }
    }
    new_student->next=cur_student;
    perv_student->next =new_student;
    return students;
}

clist* add_course(clist *courses, char *title, int number){
    clist *new_course;
    clist *cur_course ;
    clist *perv_course;
    course *crs ;/*allocate memory for a new course*/
    crs = (struct course*)malloc(sizeof(struct course));
    if(crs==NULL){
        exit(1);
    }
    /*insert course info*/
    crs->title=(char*) malloc((strlen(title)+1)*sizeof(char));
    if(crs->title==NULL){
        exit(1);
    }
    strcpy(crs->title,title);/*have to copy, unless ints a pinter*/
    crs->number=number;
    crs->students=0;
    
    new_course =(struct clist*)malloc(sizeof(struct clist));
    if(new_course==NULL){
        exit(1);
    }
    /*temp changing pointer courses*/
    cur_course = courses;
    new_course->info =crs;
    if(!courses){/*in case that the list is empty*/
        new_course->next=0;
        return new_course;
    }
    if(crs->number < courses->info->number){/*in case that the course number of the head of the list is bigger than the new course*/
        new_course->next=courses;
        return new_course;
    }
    
    while (cur_course){/*in case need to insert the course in middle location or at the end*/
        if(crs->number > cur_course->info->number) {
            perv_course=cur_course;
            cur_course = cur_course->next;
        }
        else{
            break;
        }
    }
    new_course->next=cur_course;
    perv_course->next =new_course;
    return courses;
}

void reg_student(slist *students, clist *courses, int id, int number){
    slist *new_student;
    slist *std = find_student(students,id);
    clist *crs = find_course(courses,number);
    clist *new_course=(clist*) malloc(sizeof(clist*));
    if(new_course==NULL){
        exit(1);
    }
    new_course->info=crs->info;
    
    new_student=(slist*)malloc(sizeof (slist*));
    if(new_student==NULL){
        exit(1);
    }
    new_student->info=std->info;
    insert_course(std->info->courses,number,new_course,std);/*insert the course to the student courses list*/
    insert_student(crs->info->students,id,new_student,crs);/*insert the student to the course student list*/
}

void unreg_student(slist *students, int id, int number){
    slist *std= find_student(students,id);
    clist *crs = find_course(std->info->courses,number);
    remove_student(crs,id,std);/*remove the student from the course student list*/
    remove_course(std,number,crs);/*remove the course from the student course list*/
}

void print_students(slist *students){
    if(students==0){
        printf("STUDENT LIST: EMPTY!\n");
    }
    else {
        slist *cur_student;
        clist *cur_course;
        printf("STUDENT LIST: \n");
        
        cur_student = students;
        
        while (cur_student) {
            printf("%d:%s\n", cur_student->info->id, cur_student->info->name);
            cur_course = cur_student->info->courses;
            if (cur_course == 0) {
                printf("student is not registered for courses.\n");
            } else {
                printf("courses: ");
                while (cur_course) {
                    printf("%d-%s", cur_course->info->number, cur_course->info->title);
                    if (cur_course->next != 0) {
                        printf(", ");
                    } else {
                        printf("\n");
                    }
                    cur_course = cur_course->next;

                }
            }
            cur_student = cur_student->next;
        }
    }
}

void print_courses(clist *courses){
    if(!courses){
        printf("COURSE LIST: EMPTY!\n");
    }
    else{
        clist *cur_course;
        printf("COURSE LIST:\n");
        
        cur_course=courses;
        while(cur_course){
            printf("%d:%s\n",cur_course->info->number,cur_course->info->title);
            if(!cur_course->info->students){
                printf("course has no students.\n");
            }
            else{
                slist *cur_student;
                cur_student=cur_course->info->students;
                printf("students: ");
                while(cur_student){
                    printf("%d-%s",cur_student->info->id,cur_student->info->name);
                    if(!cur_student->next){
                        printf("\n");
                    }
                    else{
                        printf(", ");
                    }
                    cur_student=cur_student->next;
                }
            }
            cur_course=cur_course->next;
        }
    }
}

void free_all(slist *sl, clist *cl){
    /*TODO check if need to free info*/
    /*free courses*/
    clist *temp_c;
    slist *cur_student;
    slist  *temp_s;
    clist *cur_course;
    cur_course=cl;
    
    while(cur_course){
        temp_c =cur_course;
        free_course(cur_course->info);/*free the course info*/
        cur_course=temp_c->next;
        /*free(temp_c->info);*/
        free(temp_c);
    }
    
    cur_student=sl;
    
    while (cur_student){
        temp_s=cur_student;
        free_student(cur_student->info);
        cur_student=cur_student->next;
        /*free(temp_s->info);*/
        free(temp_s);
    }
}

/*my functions*/
slist* find_student(slist *students,int id) {
    slist *cur_student;
    cur_student = students;
    while(cur_student->info->id!=id){
        cur_student=cur_student->next;
    }
    return cur_student;
}

clist* find_course(clist *corses,int number){
    clist *cur_course ;
    cur_course = corses;
    while (cur_course->info->number!=number){
        cur_course=cur_course->next;
    }
    return cur_course;
}

void insert_course(clist *courses,int number,clist *course,slist *student){
    clist *cur_course ;
    clist *perv_course;
    cur_course = student->info->courses;
    if(!courses){
        student->info->courses=course;
        student->info->courses->next=0;
        return ;
    }
    if(number < cur_course->info->number){/*in case that the course number of the head of the list is bigger than the new course*/
        course->next=courses;
        student->info->courses=course;
        return ;
    }
    
    while (cur_course){/*in case need to insert the course in middle location or at the end*/
        if(number > cur_course->info->number){
            perv_course=cur_course;
            cur_course=cur_course->next;
        }
        else{
            break;
        }
    }
    course->next=cur_course;
    perv_course->next =course;
}

void insert_student(slist *students,int id ,slist *student,clist *course){
    slist *cur_student ;
    slist *perv_student;
    cur_student =course->info->students;
    if(!students){
        course->info->students=student;
        course->info->students->next=0;
        return;
    }
    if(id<cur_student->info->id){
        student->next=students;
        course->info->students=student;
        return;
    }
    
    while(cur_student){
        if(id>cur_student->info->id) {
            perv_student=cur_student;
            cur_student = cur_student->next;
        }
        else{
            break;
        }
    }
    student->next=cur_student;
    perv_student->next =student;
}

void remove_course(slist *student ,int number,clist *course){
    clist *cur_course;
    clist *perv_course;
    cur_course=student->info->courses;
    
    if(number==cur_course->info->number){
        student->info->courses=cur_course->next;
        free(cur_course);
    }
    else {
        while (cur_course) {
            if (number != cur_course->info->number) {
                perv_course = cur_course;
                cur_course = cur_course->next;
            } else {
                break;
            }
        }
        perv_course->next = cur_course->next;
        free(cur_course);
    }
}

void remove_student(clist *course,int id,slist *student) {
    slist *cur_student ;
    slist *perv_student;
    cur_student = course->info->students;
    
    if(id==cur_student->info->id){
        course->info->students=cur_student->next;
        free(cur_student);
    }
    else {
        while (cur_student) {
            if (id != cur_student->info->id) {
                perv_student = cur_student;
                cur_student = cur_student->next;
            } else {
                break;
            }
        }
        perv_student->next = cur_student->next;
        free(cur_student);
    }
}

void free_course(course *course){
    slist *cur_student;
    slist *temp;
    free(course->title);/*there is no need to free course.number because it is not dynamically allocated memory*/
    
    cur_student=course->students;
    
    /*free the student list of the course*/
    while(cur_student){/*free slist, list of pointers*/
        temp=cur_student;
        cur_student=cur_student->next;
        free(temp);
    }
    free(course);
}

void free_student(student *student){
    clist *cur_course;
    clist *temp;
    free(student->name);/*there is no need to free course.number because it is not dynamically allocated memory*/
    
    cur_course=student->courses;
    
    /*free the clist of the student*/
    while(cur_course){
        temp=cur_course;
        cur_course=cur_course->next;
        free(temp);
    }
    free(student);
}
